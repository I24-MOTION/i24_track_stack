import torch
import PyNvCodec as nvc
import PytorchNvCodec as pnvc


import configparser
import torch
import socket

import os
import numpy as np
import time
import re
import glob 

import cv2
from PIL import Image

from torchvision.transforms import functional as F

from i24_logger.log_writer import logger,catch_critical

class ManagerClock:
    def __init__(self,start_ts,desired_processing_speed,framerate):
        """
        start_ts - unix timestamp
        desired_processing_speed - int (if 1, realtime processing expected, if 0 no processing speed constraint,etc)
        framerate - int nominal capture speed of camera
        """

        self.dps = desired_processing_speed
        self.framerate = framerate
        
        self.start_ts = start_ts
        self.start_t = time.time()
        
        self.target_time = self.start_ts
        
    def tick(self):
        """
        Returns the later of:
            1. max ts from previous camera timestamps + framerate
            2. the timestamp at which processing should be to maintain desired processing speed
        
        ts - list of lists of timestamps returned by loader
        """
        # flat_ts = []
        # for item in ts:
        #     flat_ts += item
        # max_ts = max(flat_ts)
        
        #max_ts = max(ts)
        
        target_ts_1 = self.target_time + 1.0/self.framerate
        
        
        elapsed_proc_time = time.time() - self.start_t
        target_ts_2 = self.start_ts + elapsed_proc_time*self.dps
        
        self.target_time = max(target_ts_1,target_ts_2)
        return self.target_time
    
class MCLoader():
    """
    Loads multiple video files in parallel with PTS timestamp decoding and 
    directory - overall file buffer
    """
    
    @catch_critical()        
    def __init__(self,directory,mapping,cam_names,ctx,resize = (1920,1080), start_time = None, Hz = 29.9):
        
        
    
        self.cam_devices = mapping


        # instead of getting individual files, sequence is a directorie (1 per camera)
        cam_sequences = {}        
        for file in os.listdir(directory):
            sequence = os.path.join(directory,file)
            if os.path.isdir(sequence):
                cam_name = re.search("P\d\dC\d\d",sequence).group(0)
                cam_sequences[cam_name] = sequence
        
        self.true_start_time = self.get_start_time(cam_names,cam_sequences)
        self.true_start_time += 1

        print("Loader says: True File Start Time: {}   User/Checkpoint Supplied Start Time: {}".format(self.true_start_time,start_time))        
        if start_time is None:
            self.start_time = self.true_start_time
        else:
            self.start_time = start_time
            
        print("Loader sent start time: {} to worker processes".format(self.start_time))
        
        # device loader is a list of lists, with list i containing all loaders for device i (hopefully in order but not well enforced by dictionary so IDK)
        self.device_loaders = [[] for i in range(torch.cuda.device_count())]
        self.device_loader_cam_names =  [[] for i in range(torch.cuda.device_count())]
        for key in cam_names:
            dev_id = self.cam_devices[key.lower().split("_")[0]]
            
            try:
                sequence = cam_sequences[key.split("_")[0]]
            except:
                sequence = cam_sequences[key.upper().split("_")[0]]
            
            loader = GPUBackendFrameGetter(sequence,dev_id,ctx,resize = resize,start_time = self.start_time, Hz = Hz)
            
            self.device_loaders[dev_id].append(loader)
            self.device_loader_cam_names[dev_id].append(sequence)
            
        
        self.returned_counter = 0
        self.target_time = self.start_time
        self.Hz = Hz
        
        for dev in self.device_loaders:
            for loader in dev:
                loader.__next__(timeout = 300)
        logger.info("Successfully got first frame from all cameras")
     
    @catch_critical()                          
    def _parse_device_mapping(self,mapping_file):
        """
        This function is likely to change in future versions. For now, config file is expected to 
        express camera device as integer e.g. p1c1=3
        :param mapping_file - (str) name of file with camera mapping
        :return dict with same information p1c1:3
        """
        mapping_file = os.path.join(os.environ["USER_CONFIG_DIRECTORY"],mapping_file)
        cp = configparser.ConfigParser()
        cp.read(mapping_file)
        mapping = dict(cp["DEFAULT"])
       
        for key in mapping.keys():
            parsed_val = int(mapping[key])
            mapping[key] = parsed_val
    
        self.cam_devices = mapping       
        
    @catch_critical()     
    def get_frames(self,tolerance = 1/60):
            self.target_time = self.start_time + self.returned_counter * 1/self.Hz
        
        # try:
            # accumulators
            frames = [[] for i in range(torch.cuda.device_count())]
            timestamps = []
            
            # advance each camera loader
            for dev_idx,this_dev_loaders in enumerate(self.device_loaders):
                for l_idx,loader in enumerate(this_dev_loaders):
                    
                    if loader.item[1] >self.target_time + tolerance: # this frame is too far ahead, do not increment
                        frame = torch.zeros(3,1080,1920,device= torch.device("cuda:{}".format(dev_idx)))
                        ts = -np.inf
                        logger.warning("Loader for sequence {} is too far ahead (target time: {},    next frame timestamp: {}) and did not increment frames. Possible dropped packet.".format(self.device_loader_cam_names[dev_idx][l_idx],self.target_time,loader.item[1]))
                        
                    else:
                        ts = -1
                        while ts < self.target_time - tolerance:
                            # advancement of at least one frame
                            frame,ts = loader.item
                            next(loader)
                        
                    frames[dev_idx].append(frame)
                    timestamps.append(ts)
                        
            # stack each accumulator list
            out = []
            for lis in frames:
                if len(lis) == 0: # occurs when no frames are mapped to a GPU
                    out.append(torch.empty(0))
                else:
                    
                    out.append(torch.stack(lis))
            #timestamps = torch.tensor([torch.tensor(item) for item in timestamps],dtype = torch.double)
            
            self.returned_counter += 1 
            return out,timestamps
    
        # except: # end of input
        #     return None, None
        
    @catch_critical()
    def get_start_time(self,cam_names,cam_sequences):
        all_ts = []
        for key in cam_names:
            gpuID = self.cam_devices[key.lower().split("_")[0]]
            
            try:
                directory = cam_sequences[key.split("_")[0]]
            except:
                directory = cam_sequences[key.upper().split("_")[0]]
                
            
            # filter out non-video_files and sort video files
            #files = os.listdir(directory)
            #files = list(filter(  (lambda f: True if ".mkv" in f else False) ,   files))
            files = glob.glob(directory+"/*.mkv")
            files.sort()
            sequence = files[0]
            #sequence = os.path.join(directory, files[0])
            
            ts  = float(sequence.split("/")[-1].split(".mkv")[0].split("_")[-1])
            
            # nvDec = nvc.PyNvDecoder(sequence,gpuID)
            # target_h, target_w = nvDec.Height(), nvDec.Width()
        
            # to_rgb = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpuID)
            # to_planar = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpuID)
        
            # cspace, crange = nvDec.ColorSpace(), nvDec.ColorRange()
            # if nvc.ColorSpace.UNSPEC == cspace:
            #     cspace = nvc.ColorSpace.BT_601
            # if nvc.ColorRange.UDEF == crange:
            #     crange = nvc.ColorRange.MPEG
            # cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)
            
            
            # pkt = nvc.PacketData()
            # rawSurface = nvDec.DecodeSingleSurface(pkt)
            # ts = pkt.pts      / 10e8          
            all_ts.append(ts)
            
        return max(all_ts)

    def __del__(self):
        try:
            for idx in range(len(self.device_loaders)):
                for loader in self.device_loaders[idx]:
                    loader.worker.kill()
        except:
            pass
            
    
class GPUBackendFrameGetter:
    def __init__(self,directory,device,ctx,buffer_size = 5,resize = (1920,1080),start_time = None, Hz = 29.9):
        
        # create shared queue
        self.queue = ctx.Queue()
        self.frame_idx = -1
        self.device = device  
        
        self.directory = directory
        # instead of a single file, pass a directory, and a start time
        self.worker = ctx.Process(target=load_queue_continuous_vpf, args=(self.queue,directory,device,buffer_size,resize,start_time,Hz),daemon = True)
        self.worker.start()   
        
        self.directory = directory        

    def __len__(self):
        """
        Description
        -----------
        Returns number of frames in the track directory
        """
        
        return 1000000
    
    
    def __next__(self,timeout = 10):
        """
        Description
        -----------
        Returns next frame and associated data unless at end of track, in which
        case returns -1 for frame num and None for frame

        Returns
        -------
        frame_num : int
            Frame index in track
        frame : tuple of (tensor,tensor,tensor)
            image, image dimensions and original image

        """
        
        
        try:
            frame = self.queue.get(timeout = timeout)
        
            ts = frame[1] 
            im = frame[0]
        except Exception as e:
            logger.error("Got timeout exception {} loading frames from {}. None frame will be returned and tracking will shut down".format(e,self.directory))
            im = None #torch.empty([6,1080,1920])
            ts = None
            
        self.item = (im,ts)
        #return im,ts
        
        # if False: #TODO - implement shutdown
        #     self.worker.terminate()
        #     self.worker.join()
        #     return None
        
def load_queue_continuous_vpf(q,directory,device,buffer_size,resize,start_time,Hz,tolerance = 1/60.0):
    
    logger.set_name("Hardware Decode Handler {}".format(device))
    
    resize = (resize[1],resize[0])
    gpuID = device
    device = torch.cuda.device("cuda:{}".format(gpuID))
    
    camera = re.search("P\d\dC\d\d",directory).group(0)
    # load mask file
    if socket.gethostname() == "quadro-cerulean":
        mask_path = "/home/derek/Documents/i24/i24_track/data/mask/{}_mask_1080.png".format(camera)
    else:
        mask_path = "/remote/i24_code/tracking/data/mask/{}_mask_1080.png".format(camera)
    mask_im = np.asarray(Image.open(mask_path))
    mask_im = torch.from_numpy(mask_im.copy()) 
    mask_im = torch.clamp(mask_im.to(gpuID).unsqueeze(0).expand(3,mask_im.shape[0],mask_im.shape[1]),min = 0, max = 1)
    logger.info("{} mask im shape: {}. Max value {}".format(mask_path,mask_im.shape,torch.max(mask_im)))
    
    # GET FIRST FILE
    # sort directory files (by timestamp)
    #files = os.listdir(directory)
    
    # filter out non-video_files and sort video files
    #files = list(filter(  (lambda f: True if ".mkv" in f else False) ,   files))
    files = glob.glob(directory+"/*.mkv")
    files.sort()
    
    # select next file that comes sequentially after last_file
    for fidx,file in enumerate(files):
        try:
            ftime = float(         file.split("_")[-1].split(".mkv")[0])
            nftime= float(files[fidx+1].split("_")[-1].split(".mkv")[0])
            if nftime >= start_time:
                break
        except:
            logger.warning("Selecting the file to load from, the last file was selected.")
            break # no next file so this file should be the one
    
    logger.debug("Loading frames from {}".format(file))
    last_file = file
    

    
    returned_counter = 0
    while True:
        
        #file = os.path.join(directory,file)
        
        # initialize Decoder object
        nvDec = nvc.PyNvDecoder(file, gpuID)
        target_h, target_w = nvDec.Height(), nvDec.Width()
    
        to_rgb = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpuID)
        to_planar = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpuID)
    
        cspace, crange = nvDec.ColorSpace(), nvDec.ColorRange()
        if nvc.ColorSpace.UNSPEC == cspace:
            cspace = nvc.ColorSpace.BT_601
        if nvc.ColorRange.UDEF == crange:
            crange = nvc.ColorRange.MPEG
        cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)
        
        # get first frame
        pkt = nvc.PacketData()                    
        rawSurface = nvDec.DecodeSingleSurface(pkt)
        ts = pkt.pts /10e8
        
        # get frames from one file
        while True:
            if q.qsize() < buffer_size:                
                
                target_time = start_time + returned_counter * 1/Hz
                
                c = 0
                while ts + tolerance < target_time:
                    pkt = nvc.PacketData()                    
                    rawSurface = nvDec.DecodeSingleSurface(pkt)
                    ts = pkt.pts /10e8
                    
                    if rawSurface.Empty():
                        break
               
                
                # Obtain NV12 decoded surface from decoder;
                #raw_surface = nvDec.DecodeSingleSurface(pkt)
                if rawSurface.Empty():
                    logger.debug("raw surface empty")
                    break
    
                # Convert to RGB interleaved;
                rgb_byte = to_rgb.Execute(rawSurface, cc_ctx)
            
                # Convert to RGB planar because that's what to_tensor + normalize are doing;
                rgb_planar = to_planar.Execute(rgb_byte, cc_ctx)
                
                # likewise, end of video file
                if rgb_planar.Empty():
                    break
                rgb_planar = rgb_planar.Clone(gpuID)


                # Create torch tensor from it and reshape because
                # pnvc.makefromDevicePtrUint8 creates just a chunk of CUDA memory
                # and then copies data from plane pointer to allocated chunk;
                surfPlane = rgb_planar.PlanePtr()
                surface_tensor = pnvc.makefromDevicePtrUint8(surfPlane.GpuMem(), surfPlane.Width(), surfPlane.Height(), surfPlane.Pitch(), surfPlane.ElemSize())
                surface_tensor.resize_(3, target_h,target_w)
                
                # apply mask
                surface_tensor = surface_tensor* mask_im.expand(surface_tensor.shape)
                
                try:
                    surface_tensor = torch.nn.functional.interpolate(surface_tensor.unsqueeze(0),resize).squeeze(0)
                except:
                    raise Exception("Surface tensor shape:{} --- resize shape: {}".format(surface_tensor.shape,resize))
            
                # This is optional and depends on what you NN expects to take as input
                # Normalize to range desired by NN. Originally it's 
                surface_tensor = surface_tensor.type(dtype=torch.cuda.FloatTensor)/255.0
                 
                if False: # write frame so we can get all inspecty on it
                    outim = (surface_tensor.permute(1,2,0).clone().cpu().data.numpy()*255).astype(np.uint8)
                    cv2.imwrite("/remote/home/glouded/phantom_car_test/{}_{}.png".format(camera,str(returned_counter).zfill(4)), outim)
                
                # apply normalization
                surface_tensor = F.normalize(surface_tensor,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                
                
                
                
                frame = (surface_tensor,ts)
                
                returned_counter += 1
                q.put(frame)
            
        
        logger.debug("Finished handling frames from {}".format(file))

            
        ### Get next file if there is one 
        # sort directory files (by timestamp)
        # files = os.listdir(directory)
        
        # # filter out non-video_files and sort video files
        # files = list(filter(  (lambda f: True if ".mkv" in f else False) ,   files))
        # files.sort()
        files = glob.glob(directory+"/*.mkv")
        files.sort()
        
        logger.debug("Available files {}".format(files))
        
        # select next file that comes sequentially after last_file
        NEXTFILE = False
        for file in files:
            if file > last_file:
                last_file = file
                NEXTFILE = True           
                logger.debug("Loading frames from {}".format(file))
                break

        
        if not NEXTFILE:
            logger.warning("Loader {} ran out of input in directory {}.".format(gpuID,directory))
            break
            #raise Exception("Reached last file for directory {}".format(directory))
            