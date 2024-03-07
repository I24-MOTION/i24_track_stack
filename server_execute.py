import torch.multiprocessing as mp
import torch

import socket
import _pickle as pickle
import numpy as np
import os,shutil
from sys import exit
import time
import signal 
import warnings

#os.environ["USER_CONFIG_DIRECTORY"] = "/home/derek/Documents/i24/i24_track/config/lambda_cerulean_eval2"
if socket.gethostname() == 'lambda-cerulean':    
    os.environ["USER_CONFIG_DIRECTORY"] = "/home/worklab/Documents/i24_common_sim/config/lambda_cerulean_batch_6"
from i24_logger.log_writer         import logger,catch_critical,log_warnings
mp.set_sharing_strategy('file_system')


# relative imports
from src.util.bbox                 import state_nms,estimate_ts_offsets
from src.util.misc                 import plot_scene,colors,Timer
from src.track.tracker             import get_Tracker, get_Associator
from src.track.trackstate          import TrackierState as TrackState
from src.detect.pipeline           import get_Pipeline
from src.scene.devicemap           import get_DeviceMap
from src.detect.devicebank         import DeviceBank
from src.db_write                  import WriteWrapperConf as WriteWrapper

from src.load.gpu_load_multi_presynced       import MCLoader

# custom package imports
from i24_configparse               import parse_cfg,parse_delimited
from i24_rcs                       import I24_RCS


def __process_entry__(cams=[], hg_file = '', vid_dir='', ing_id='', track_id='', end_time=0,start_time=0):
    p = TrackingProcess(cams = cams,hg_file = hg_file, vid_dir = vid_dir, ing_id = ing_id, track_id = track_id,end_time=end_time,start_time=start_time)
    signal.signal(signal.SIGINT, p.sigint_handler)
    signal.signal(signal.SIGUSR1, p.sigusr_handler)
    p.main()
    
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda:0')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))








class TrackingProcess:
    
    def __init__(self,cams=[],
                 vid_dir='', 
                 hg_file = '',
                 ing_id='',
                 track_id='', 
                 end_time=0, 
                 start_time=0 ):
        """
        Set up persistent process variables here
        """
        mp.set_sharing_strategy('file_system')
        force_cudnn_initialization()

        self.run = True
        ctx = mp.get_context('spawn')
        
        import gc
        torch.cuda.empty_cache()
        gc.collect()

        
        from i24_logger.log_writer         import logger,catch_critical,log_warnings
        self.logger = logger
        self.logger.set_name("Tracking Main")
        logger.info("Main process PID: {}".format(os.getpid()))
        
        #%% run settings    
        self.tm = Timer()
        self.tm.split("Init")
        
        par = {"cams":cams,"vid_dir":vid_dir,"ing_id":ing_id,"track_id":track_id,"end_time":end_time,"start_time":start_time}
        self.logger.info("Got input params: {}".format(par),extra = par)
        run_config = "execute.config"       
        #mask = ["p2c1", "p2c3","p2c5","p3c1"] #["p46c01","p46c02", "p46c03", "p46c04", "p46c05","p46c06"]
        self.mask = None
        
        # load parameters
        params = parse_cfg("TRACK_CONFIG_SECTION",
                           cfg_name=run_config, SCHEMA=False)
        
        # default input directory
        in_dir = params.input_directory #os.path.join(params.input_directory,track_id)
        self.checkpoint_dir = in_dir
        
        self.max_crops = params.max_crops
        
        # default hostname
        hostname = socket.gethostname()
        self.hostname = hostname
        
        # this is a spoof to pretend to be a particular video node on my local desktop machine
        if socket.gethostname() == 'lambda-cerulean':    
            hostname = "videonode2"
        
        # default camera list
        cam_assignment =  params.cam_assignment
        ret = parse_delimited(cam_assignment, "name")
        out = filter(lambda x: ret[x].host == hostname, ret.keys())
        include_camera_list = list(out)
        
        if len(include_camera_list) == 0:
            include_camera_list = None
        
        # default collection name
        self.collection_overwrite = "garbage_dump"
        
        ##### overwrite with stuff kwargs
        
        # if camera list isn't specified as an argument, get it from params
        if len(cams) > 0:
            include_camera_list = cams
        
        if include_camera_list is None:
            self.run = False
            return 
        
        priorities = [ret[key].priority for key in include_camera_list]
        
        # get input video directory. If  not specified in arg1, get it from execute.config
        if len(vid_dir) > 0:
            in_dir = vid_dir# os.path.join(vid_dir,track_id)
        self.checkpoint_dir = in_dir

        
        if len(track_id) > 0:
            self.collection_overwrite = track_id
        
        
        self.hg = I24_RCS(hg_file,downsample = 2)
        
        
        # TODO trim include_camera_list
        if True:
            removals = []
            for camera in include_camera_list:
                hit = False
                for key in list(self.hg.correspondence.keys()):
                    if camera.upper() in key.upper():
                       hit = True
                       break
                if not hit:
                    removals.append(camera)
                    logger.warning("Camera {} specified in cams input to Tracking, but no camera correspondence exists in homography object.".format(camera))
            for removal in removals:
                include_camera_list.remove(removal)
                    
        
        
        # fill missing
        for p in range(1,41):
            for c in range(1,7):
            # if "P{}C{}_WB".format(str(p).zfill(2)) not in self.hg.correspondence.keys() and "P{}C03_EB".format(str(p).zfill(2)) in self.hg.correspondence.keys():
            #     self.hg.correspondence["P{}C03_WB".format(str(p).zfill(2))] = self.hg.correspondence["P{}C03_EB".format(str(p).zfill(2))]
            #     print("Fill")
            # if "P{}C04_EB".format(str(p).zfill(2)) not in self.hg.correspondence.keys() and "P{}C04_WB".format(str(p).zfill(2)) in self.hg.correspondence.keys():
            #     self.hg.correspondence["P{}C04_EB".format(str(p).zfill(2))] = self.hg.correspondence["P{}C04_WB".format(str(p).zfill(2))]
            #     print("Fill")
            
                sideA = "P{}C{}_EB".format(str(p).zfill(2),str(c).zfill(2))
                sideB = "P{}C{}_WB".format(str(p).zfill(2),str(c).zfill(2))
                if sideA in self.hg.correspondence.keys() and sideB not in self.hg.correspondence.keys():
                    self.hg.correspondence[sideB] = self.hg.correspondence[sideA].copy()
                    logger.info("Using {} correspndence for {} as none exists.".format(sideA,sideB))
                elif sideB in self.hg.correspondence.keys() and sideA not in self.hg.correspondence.keys():
                    self.hg.correspondence[sideA] = self.hg.correspondence[sideB].copy()
                    logger.info("Using {} correspndence for {} as none exists.".format(sideB,sideA))
          
       
                
                
        # intialize DeviceMap
        self.dmap = get_DeviceMap(params.device_map, self.hg, camera_list = include_camera_list, camera_priorities = priorities)
        dmap_devices = list(set(self.dmap.cam_devices))
        dmap_devices.sort()
        params.cuda_devices = dmap_devices
        
        print(self.dmap.cam_names,self.dmap.cam_devices)
        
        # intialize empty TrackState Object
        self.tstate = TrackState()
        target_time = None
        
        # load checkpoint
        target_time,self.tstate,self.collection_overwrite = self.load_checkpoint(target_time,self.tstate,self.collection_overwrite)
        
        if (target_time is None and start_time > 0) or (target_time is not None and target_time < start_time):
            target_time = start_time
            
        print("Collection Name: {}    Track ID:  {}   Video Path: {}   Checkpoint Path: {}".format(self.collection_overwrite,track_id,in_dir, self.checkpoint_dir))
        
        # get frame handlers
        self.loader = MCLoader(in_dir,self.dmap.cam_devices_dict,self.dmap.cam_names, ctx,start_time = target_time,Hz = params.nominal_framerate,hg_file = hg_file)
        self.max_ts = self.loader.start_time
        self.start_ts = self.loader.true_start_time
        
        
        logger.info("HG start time: {}, Tracking start time: {}  ---- difference is {:.2}sec.".format(self.hg.hg_start_time,self.start_ts,self.hg.hg_start_time-self.start_ts))
        self.hg.hg_start_time -= self.start_ts # this ensures that the correct bin will be indexed (e.g. if hg start time is 12 and video start ts is 13, a ts_trunc of 0 (e.g. video ts = 13) should index the hg time 1 second after the start so hg_start_ts needs to be -1
        
        self.logger.debug("Initialized {} loader processes.".format(len(self.loader.device_loaders)))
        print("In main loop, max_ts = {} and start_ts = {}".format(self.max_ts,self.start_ts))
        
        

        
        if params.track:
            # initialize pipelines
            pipelines = params.pipelines
            pipelines = [[get_Pipeline(item, self.hg) for item in pipelines] for _ in params.cuda_devices]
            associators = params.associators
            self.associators = [get_Associator(item) for item in associators]
            
            # initialize tracker
            self.tracker = get_Tracker(params.tracker)
            
            if params.trim_extents:
                minx = torch.min(self.dmap.cam_extents[:,0]).item()
                maxx = torch.max(self.dmap.cam_extents[:,1]).item()
                print(minx,maxx)
                self.tracker.fov = [minx,maxx]
                # get max and min of all camera extents
                # overwrite tracker FOV extents
            
            # add Associate function to each pipeline
            # for i in range(len(pipelines)):
            #     assoc = associators[i]
            #     pipelines[i].associate = associators[i]
            
            # initialize DetectorBank
            self.dbank = DeviceBank(params.cuda_devices, pipelines, self.dmap.gpu_cam_names, ctx)
            
        # initialize DBWriter object
        if params.write_db:
            self.dbw = WriteWrapper(collection_overwrite = self.collection_overwrite,server_id = hostname, mm_offset = self.hg.MM_offset)
        else:
            self.dbw = []

        # cache params
        self.params = params
        
        self.end_time = (np.inf if end_time <= 0 else end_time)
        

    # @log_warnings()
    # def parse_cfg_wrapper(self,run_config):
    #     params = parse_cfg("TRACK_CONFIG_SECTION",
    #                        cfg_name=run_config, SCHEMA=False)
    #     return params
    
    
    
    @catch_critical()
    def checkpoint(self):
        """
        Saves the trackstate and next target_time as a pickled object such that the 
        state of tracker can be reloaded for no loss in tracking progress
        
        :param   tstate - TrackState object
        :param   next_target_time - float
        :return  None
        """
        base_name = self.hostname + "_" + self.collection_overwrite + ".cpkl"
        save_file = os.path.join(self.checkpoint_dir,base_name)
        with open(save_file,"wb") as f:
            pickle.dump([self.max_ts,self.tstate,self.collection_overwrite],f)
        logger.debug("Checkpointed TrackState object, time:{}s".format(self.max_ts))
    
            
    @catch_critical()
    def load_checkpoint(self,target_time,tstate,collection_overwrite):
        """
        Loads the trackstate and next target_time from pickled object such that the 
        state of tracker can be reloaded for no loss in tracking progress. Requires 
        input time and tstate such that objects can be naively passed through if no 
        save file exists
        
        :param   tstate - TrackState object
        :param   next_target_time - float
        :return  None
        """  
        base_name = self.hostname + "_" + self.collection_overwrite + ".cpkl"
        save_file = os.path.join(self.checkpoint_dir,base_name)
        
        if os.path.exists(save_file):
            with open(save_file,"rb") as f:
                target_time,tstate,collection_overwrite = pickle.load(f)
            
            logger.debug("Loaded checkpointed TrackState with {} objects, time:{}s".format(len(tstate),target_time))
            
        else:
            logger.debug("No checkpoint file exists, starting tracking from max min video timestamp")
            
        return target_time,tstate,collection_overwrite
            
    def sigint_handler(self,sig,frame):
        self.run = False
        self.logger.warning("Either SIGINT or KeyboardInterrupt recieved. Initiating soft shutdown")

        #self.checkpoint()
        
        # clean up subprocesses
        del self.dbank,
        torch.cuda.empty_cache()
        print("Deleted device handlers")
        time.sleep(5)
        

        del self.loader
        torch.cuda.empty_cache()
        print("Deleted frame loaders")

        del self.dbw

        
        self.logger.debug("Soft Shutdown complete. All processes should be terminated")
        self.logger.debug("Exiting in 3 seconds")
        time.sleep(3)
        exit()
        
        raise KeyboardInterrupt("Re-raising error after soft shutdown.")
    
    def sigusr_handler(self,sig,frame):
        self.logger.warning("SIGUSR recieved. Flushing active objects then raising SIGINT")
        
        logger.debug("Keyboard Interrupt recieved. Initializing soft shutdown")

        residual_objects,COD = self.tracker.flush(self.tstate)
        try:
            self.dbw.insert(residual_objects,cause_of_death = COD,time_offset = self.start_ts)
            self.logger.info("Flushed all active objects to database")
        except:
            logger.warning("Failed to flush active objects at end of tracking. Is write_db = False?")
        
        self.sigint_handler(sig,frame)
    
    
    def main(self):  
          
        
        
        #%% Main Processing Loop
        if self.run:
            
            # initialize processing sync clock
            fps = 0
            frames_processed = 0
            term_objects = 0
            
                        
            start_time = time.time()
            
            self.logger.debug("Initialization Complete. Starting tracking at {}s".format(start_time))
        
            # readout headers
            
            
            time_dict = {}
            
            try:
                while self.max_ts <self.end_time:
                        
        
                    
                    if self.params.track: # shortout actual processing
                    
                       
                    
                        # select pipeline for this frame
                        pidx = frames_processed % len(self.params.pipeline_pattern)
                        pipeline_idx = self.params.pipeline_pattern[pidx]
                        if len(self.tstate) > self.max_crops:
                            pipeline_idx = 0
                        
                        if pipeline_idx != -1: # -1 means skip frame
                            
                            # get next frames and timestamps
                            self.tm.split("Get Frames")
                            previous_max_ts = self.max_ts
                            try:
                                frames, timestamps = self.loader.get_frames()
                                self.max_ts = max(timestamps)
                            except Exception as e:
                                logger.warning("Exiting main loop beacuse error thrown in get_frames() or max_ts calculation: {}".format(e))
                                self.max_ts = previous_max_ts + 0.1
                                break # out of input
                            #print([len(f) for f in frames])
                            #[print(f.shape,f.device,"||") for f in frames]
        
                            if self.max_ts < -20:
                                self.max_ts += 0.1
                                logger.warning("Max ts is -inf, which likely means no frames are available. Resetting max_ts to 0.1s past previous max ({})s and loading next frame".format(self.max_ts))
                                continue
                            #print(frames_processed,timestamps[0],target_time) # now we expect almost an exactly 30 fps framerate and exactly 30 fps target framerate
                            
                            if frames is None:
                                logger.warning("frames is None. Tracker is shutting down")
                                break #out of input
                            ts_trunc = [item - self.start_ts for item in timestamps]
                            
                            for t in range(len(ts_trunc)):
                                time_dict[self.dmap.cam_names[t]] = ts_trunc[t]
                                
                            # for obj in initializations:
                            #     obj["timestamp"] -= start_ts
                            initializations = None
                                
                            ### WARNING! TIME ERROR INJECTION
                            # ts_trunc[3] += 0.05
                            # ts_trunc[10] += .1
                            
                            self.tm.split("Predict")
                            camera_idxs, device_idxs, obj_times, selected_obj_idxs = self.dmap(self.tstate, ts_trunc)
                            obj_ids, priors, _ = self.tracker.preprocess(
                                self.tstate, obj_times)
                            
                            # slice only objects we care to pass to DeviceBank on this set of frames
                            # DEREK NOTE may run into trouble here since dmap and preprocess implicitly relies on the list ordering of tstate
                            if len(obj_ids) > 0:
                                obj_ids     =     obj_ids[selected_obj_idxs]
                                priors      =      priors[selected_obj_idxs,:]
                                device_idxs = device_idxs[selected_obj_idxs]
                                camera_idxs = camera_idxs[selected_obj_idxs]
                            
                            # prep input stack by grouping priors by gpu
                            self.tm.split("Map")
                            cam_idx_names = None  # map idxs to names here
                            prior_stack = self.dmap.route_objects(
                                obj_ids, priors, device_idxs, camera_idxs, run_device_ids=self.params.cuda_devices)
                        
                        
                            #print("Prior Stack Length: {}".format(len(prior_stack)))
                            if False:#  and frames_processed == 173:
                                # test on a single on-process pipeline
                                self.pipelines[pipeline_idx].set_device(1)
                                self.pipelines[pipeline_idx].set_cam_names(self.dmap.gpu_cam_names[1])
                                test = self.pipelines[pipeline_idx](frames[1],prior_stack[1],time_dict)

                    
                        
                            # get detections
                            self.tm.split("Detect {}".format(pipeline_idx),SYNC = True)
                            detections, confs, classes, detection_cam_names, associations = self.dbank(
                                prior_stack, frames, pipeline_idx=pipeline_idx,times = time_dict)
                            
                            detections = detections.float()
                            confs = confs.float()
                            
                            # THIS MAY BE SLOW SINCE ITS DOUBLE INDEXING
                            detection_times = torch.tensor(
                                [ts_trunc[self.dmap.cam_idxs[cam_name]] for cam_name in detection_cam_names])
                            
                            # now that times are saved, we don't need device referencing any more
                            # so we can append directions to cam_names
                            #detection_cam_names = [detection_cam_names[i] + ("_eb" if detections[i,5] == 1 else "_wb") for i in range(len(detection_cam_names))]
        
                            
                            if True and pipeline_idx == 0 and len(detection_cam_names) > 0:
                                keep = self.dmap.filter_by_extents(detections,detection_cam_names)
                                detections = detections[keep,:]
                                confs = confs[keep]
                                classes = classes[keep]
                                detection_times = detection_times[keep]
                                detection_cam_names = [detection_cam_names[_] for _ in keep]
                              
                            # filter out any detections with a bad timestamp 
                            if len(detection_times) > 0:
                                keep = torch.where(detection_times > -10, 1, 0).nonzero().squeeze()
                                detections = detections[keep,:]
                                confs = confs[keep]
                                classes = classes[keep]
                                detection_times = detection_times[keep]
                                detection_cam_names = [detection_cam_names[_] for _ in keep]
                                
                            if len(detections) == 1:
                                detections = detections.view(-1,detections.shape[-1])
                            
                            # Association
                            self.tm.split("Associate",SYNC = True)
                            if pipeline_idx == 0:
                                detections_orig = detections.clone()
                                if True and len(detections) > 0:
                                    # do nms across all device batches to remove dups
                                    #space_new = hg.state_to_space(detections)
                                    keep = state_nms(detections,confs)
                                    detections = detections[keep,:]
                                    classes = classes[keep]
                                    confs = confs[keep]
                                    detection_times = detection_times[keep]
                                
                                # overwrite associations here
                                associations = self.associators[0](obj_ids,priors,detections,self.hg)
                    
                            self.tm.split("Postprocess")
                            terminated_objects,cause_of_death = self.tracker.postprocess(
                                detections, detection_times, classes, confs, associations, self.tstate, hg = self.hg,measurement_idx =0)
                            term_objects += len(terminated_objects)
                
                            self.tm.split("Write DB")
                            if self.params.write_db:
                                self.dbw.insert(terminated_objects,cause_of_death = cause_of_death,time_offset = self.start_ts)
                            #print("Active Trajectories: {}  Terminated Trajectories: {}   Documents in database: {}".format(len(tstate),len(terminated_objects),len(dbw)))
                
            
                    # optionally, plot outputs
                    if self.params.plot:
                        self.tm.split("Plot")
                        #detections = None
                        priors = None
                        plot_scene(self.tstate, 
                                   frames, 
                                   ts_trunc, 
                                   self.dmap.gpu_cam_names,
                                   self.hg, 
                                   colors,
                                   extents=self.dmap.cam_extents_dict, 
                                   mask=self.mask,
                                   fr_num = frames_processed,
                                   detections = detections,
                                   priors = priors,
                                   save_crops_dir=None)
            
                    
                    # text readout update
                    self.tm.split("Bookkeeping")
                    rr = (max(timestamps) - self.loader.start_time)/(time.time() - start_time) 
                    fps = frames_processed / (time.time()-start_time)
                    max_dev = max(timestamps) - min(timestamps)
                    est_finish = (self.end_time - max(timestamps)) / (rr+ 1e-08)
                    if frames_processed % 50 == 0:
                        print("\n\nFrame:     Wall Time:       Since Start:        FPS:        Realtime Ratio:    Sync Time since start:     Max ts dev:     Active Objects:    Term Objects:    Estimated time to finish:")
                    print(      "{}         {:.1f} s        {:.3f} s        {:.2f}          {:.3f}               {:.3f} s                    {:.3f} s             {}              {}               {:.1f} s".format(frames_processed,time.time(), time.time() - start_time,fps,rr,self.max_ts-self.start_ts, max_dev, len(self.tstate), term_objects,est_finish))
                    #print(      "\r{}         {:.1f}s        {:.3f}s        {:.2f}          {:.3f}               {:.3f}                    {:.3f}              {}               {}          {:.1f}s".format(frames_processed,time.time(), time.time() - start_time,fps,rr,self.max_ts-self.start_ts, max_dev, len(self.tstate), term_objects,est_finish), end='\r', flush=True)

                    # get next target time
                    #target_time = clock.tick()
                    frames_processed += 1
                    
            
                    if (frames_processed) % 50 == 0:
                        metrics = {
                            "frame bps": fps,
                            "frame batches processed":frames_processed,
                            "run time":time.time() - start_time,
                            "scene time processed":max(timestamps) - self.start_ts,
                            "active objects":len(self.tstate),
                            "total terminated objects":term_objects,
                            "est_finish_time":min(est_finish,100000)
                            }
                        self.logger.info("Tracking Status Log",extra = metrics)
                        self.logger.info("Time Utilization: {}".format(self.tm),extra = self.tm.bins())
                        
                    # if self.params.checkpoint and frames_processed % 500 == 0:
                    #     self.checkpoint()
               
                    if len(self.tstate) > self.params.kill_count:
                        info = {"active objects":len(self.tstate)}
                        self.logger.critical("Number of active objects {} has exceeded kill count {}. Raising sigusr handler".format(len(self.tstate),self.params.kill_count),extra = info)
                        self.sigusr_handler(None,None)
                        self.logger.debug("After sigusr in kill count block. This code probably shouldn't be reached")
                
            # except KeyboardInterrupt:
                
            #     if self.run:
            #         print("caught keyboard interrupt and sending to main process")                
            #         os.kill(os.getpid(), signal.SIGUSR1)
            #     else:
            #         raise KeyboardInterrupt()
                
                
            except Exception as e:
                
                logger.warning("{} thrown. Checkpointing..".format(e))
                
                self.checkpoint()
                raise e
                
                
             
                
            # If we get here, code has finished over available time range without exception
            # So we should checkpoint the active objects and shut down without flushing objects
            #checkpoint(target_time,tstate,collection_overwrite)
            self.logger.info("Finished tracking over input time range. Shutting down.")
            
            if True: # Flush tracker objects
                residual_objects,COD = self.tracker.flush(self.tstate)
                self.dbw.insert(residual_objects,cause_of_death = COD,time_offset = self.start_ts)
                self.logger.info("Flushed all active objects to database",extra = metrics)
    
            # if collection_overwrite is not None:
            #     # cache settings in new folder
            #     cache_dir = "./data/config_cache/{}".format(collection_overwrite)
            #     #os.mkdir(cache_dir)
            #     shutil.copytree(os.environ["USER_CONFIG_DIRECTORY"],cache_dir)    
            #     logger.debug("Cached run settings in {}".format(cache_dir))
            
            # self.checkpoint()
            
            # delete database writer
            del self.dbw
            self.logger.info("Deleted database writer. ")
            
            return fps
        else:
            self.logger.debug("No cameras assigned, shutting down.")
            
     
if __name__ == "__main__":
    track_id = ''
    vid_dir = '/home/worklab/Data/batch_6'
    hg_file  = "/home/worklab/Documents/temp_hg_files_for_dev/hg_batch6_test.cpkl"


    if socket.gethostname() == "auxprocess1" or socket.gethostname() == "devvideo1":
        #track_id = "633c5e8bfc34583315cd6bed"
        #vid_dir = "/data/video/current/{}".format(track_id)
       
        track_id = "637443758b5b68fc4fd40c76"
        vid_dir = "/data/video/current/{}".format(track_id)
        track_id = "650210b0069d4dc9ee0877ce" # temp to not overwrite data
        
    __process_entry__(hg_file = hg_file, vid_dir = vid_dir,track_id=track_id,cams=["P11C06","P08C06","P09C06","P10C06","P13C06","P12C06","P11C03","P08C04","P09C03","P10C03","P13C03","P12C03"])
    
    # if True:
    #     import cv2
    #     import os
    #     import requests
    #     from datetime import datetime

    #     def im_to_vid(directory,name = "video",push_to_dashboard = False): 
    #         all_files = os.listdir(directory)
    #         all_files.sort()
    #         for filename in all_files:
    #             filename = os.path.join(directory, filename)
    #             img = cv2.imread(filename)
    #             height, width, layers = img.shape
    #             size = (width,height)
    #             break
            
    #         n = 0
            
    #         now = datetime.now()
    #         now = now.strftime("%Y-%m-%d_%H-%M-%S")
    #         f_name = os.path.join("/home/derek/Desktop",'{}_{}.mp4'.format(now,name))
    #         temp_name =  os.path.join("/home/derek/Desktop",'temp.mp4')
            
    #         out = cv2.VideoWriter(temp_name,cv2.VideoWriter_fourcc(*'mp4v'), 8, size)
             
    #         for filename in all_files:
    #             filename = os.path.join(directory, filename)
    #             img = cv2.imread(filename)
    #             out.write(img)
    #             print("Wrote frame {}".format(n))
    #             n += 1
                
    #             # if n > 30:
    #             #     break
    #         out.release()
            
    #         os.system("/usr/bin/ffmpeg -i {} -vcodec libx264 {}".format(temp_name,f_name))
            
    #         if push_to_dashboard:
                
                
    #             #snow = now.strftime("%Y-%m-%d_%H-%M-%S")
    #             url = 'http://viz-dev.isis.vanderbilt.edu:5991/upload?type=boxes_raw'
    #             files = {'upload_file': open(f_name,'rb')}
    #             ret = requests.post(url, files=files)
    #             print(f_name)
    #             print(ret)
    #             if ret.status_code == 200:
    #                 print('Uploaded!')
                    
    #     file = "/home/derek/Desktop/temp_frames"


            
    #     #file  = '/home/worklab/Desktop/temp'
    #     im_to_vid(file,name = "P08-P011",push_to_dashboard = True)
