from i24_configparse import parse_cfg
from i24_logger.log_writer import catch_critical,logger
import configparser
import torch
import re
import os
    


def get_DeviceMap(name,camera_list = None, camera_priorities = []):
    """
    getter function that takes a string (class name) input and returns an instance
    of the named class
    """
    if name == "DeviceMap":
        dmap = DeviceMap(camera_list = camera_list,camera_priorities=camera_priorities)
    elif name == "HeuristicDeviceMap":
        dmap = HeuristicDeviceMap(camera_list = camera_list, camera_priorities = camera_priorities)
    else:
        raise NotImplementedError("No DeviceMap child class named {}".format(name))
    
    return dmap


    
class DeviceMap():
    """
    DeviceMap objects have two main functions:            
        1.) camera mapping - given a TrackState object, and frame times for each camera
            takes a TrackState and a frame_time from each camera, and outputs a 
            set of object - camera query pairs. This set may be underfull
            (not every object is queried) or overfull (some objects are queried more than once).
            
        2.) GPU mapping - map_devices function returns manages the "pointers"
            to GPU-side frames, returning the correct GPU device and frame index for each camera
            
            Together, these two functions create a set of object queries indexing frames per GPU, which
            can then be passed to the DetectorBank to split across devices and perform any cropping etc.
    """
    
    @catch_critical()
    def __init__(self,camera_list = None, camera_priorities = []):
        
        # load config
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)

        # load self.cam_extents
        self._parse_cameras(self.camera_extents_file, camera_list = camera_list)
        
        # convert camera extents to tensor
        cam_names = []
        extents = []
        [(cam_names.append(key),extents.append(self.cam_extents[key])) for key in self.cam_extents.keys()]
        
        # DEREK TODO does this help
        #cam_names.sort()
        
        self.cam_names_extended = cam_names
        
        self.cam_names = []
        for n in self.cam_names_extended:
            n_trunc = n.split("_")[0]
            if n_trunc not in self.cam_names:
                self.cam_names.append(n_trunc)
        
        self.cam_extents_dict = self.cam_extents.copy()
        self.cam_extents = torch.tensor(extents)
        
        expansion_map = []
        for name in self.cam_names_extended:
            idx = self.cam_names.index(name.split("_")[0])
            expansion_map.append(idx)
        self.cam_expansion_map = expansion_map
        
        # invert  cam_names into dict
        self.cam_idxs = {}
        for i in range(len(self.cam_names)):
            self.cam_idxs[self.cam_names[i]] = i
        # load self.cam_devices
        if len(camera_priorities) == len(camera_list):
            self._priority_assign_devices(camera_list,camera_priorities)
            self.cam_devices_dict = self.cam_devices
        else:
            self._parse_device_mapping(self.camera_mapping_file)
        self.cam_devices = [self.cam_devices[cam_name.split("_")[0]] for cam_name in self.cam_names]
        
        
        # note that self.cam_names is THE ordering of cameras, and all other camera orderings should be relative to this ["p1c1","p1c2", etc]
        # self.cam_devices[i] is THE ordering of cameras per GPU (list of lists) [0,0,0,1,1,1 etc.]
        
        # and  self.gpu_cam_names contains device index for camera i in self.cam_ids [[p1c1,p1c2],[p1c3,p1c4] etc]
        self.gpu_cam_names = [[] for i in range(torch.cuda.device_count())]
        [self.gpu_cam_names[self.cam_devices[i]].append(self.cam_names[i]) for i in range(len(self.cam_names))]
        
        # lastly, we create cam_gpu_idx, which is a tensor with one row for each camera
        # where row i contains (gpu_idx for cam i, cam_frame_idx for cam i)
        self.cam_gpu_idx = torch.empty(len(self.cam_names),2)
        for i in range(len(self.cam_names)):
            self.cam_gpu_idx[i,0] = self.cam_devices[i]
            for j,name in enumerate(self.gpu_cam_names[self.cam_devices[i]]):
                if name == self.cam_names[i]:
                    self.cam_gpu_idx[i,1] = j
                    
        
                    
                
    @catch_critical()
    def _parse_cameras(self,extents_file, camera_list = None):
        """
        This function is likely to change in future versions. For now, config file is expected to 
        express camera range as minx,miny,maxx,maxy e.g. p1c1=100,-10,400,120
        :param extents_file - (str) name of file with camera extents
        :return dict with same information in list form p1c1:[100,-10,400,120]
        """
        extents_file = os.path.join(os.environ["USER_CONFIG_DIRECTORY"],extents_file)
        cp = configparser.ConfigParser()
        cp.read(extents_file)
        extents = dict(cp["DEFAULT"])
       
        removals = []
        for key in extents.keys():
            try:
                parsed_val = [int(item) for item in extents[key].split(",")]
                extents[key] = parsed_val

            except ValueError: # adding a $ to the end of each camera to be excluded will trigger this error and thus the camera will not be included
                removals.append(key)
                continue
            
            if camera_list is not None:
                base_key = key.split("_")[0]
                if base_key.upper() not in camera_list:
                    removals.append(key)
            
        for rem in removals:
            extents.pop(rem)
    
        self.cam_extents = extents
     
    @catch_critical()
    def _priority_assign_devices(self,camera_list,priorities):
        n_devices = torch.cuda.device_count()
        
        devices = [[] for _ in range(n_devices)]
        d_idx = 1
        for pri in [1,2,3]:
            for c_idx,cam in enumerate(camera_list):
                if priorities[c_idx] == pri:
                    devices[d_idx].append(cam.lower())
                    d_idx  = (d_idx+1)%n_devices
        
        # ravel lists into a single dict
        ravel = {}
        for d_idx in range(len(devices)):
            for cam in devices[d_idx]:
                ravel[cam] = d_idx
        self.cam_devices = ravel
        
        
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
       
        new_mapping = {}
        for key in mapping.keys():
            parsed_val = int(mapping[key])
            new_mapping[key] = parsed_val
            #wb_key = key + "_wb"
            #new_mapping[wb_key] = parsed_val
            #eb_key = key + "_eb"
            #new_mapping[eb_key] = parsed_val
    
        self.cam_devices = new_mapping       
    
    def map_cameras(self):
        raise NotImplementedError
    
    def map_devices(self,cam_map):
        """
        :param cameras - list of camera names of size n
        :return gpus - tensor of GPU IDs (int) for each of n input cameras
        """
        if len(cam_map) == 0:
            return []
        
        
        
        gpu_map = torch.tensor([self.cam_devices[camera] for camera in cam_map])
        return gpu_map
    
    @catch_critical()
    def __call__(self,tstate,ts):
        cam_map,obj_times,keep = self.map_cameras(tstate,ts)
        gpu_map = self.map_devices(cam_map)
        
        # get times
        
        return cam_map,gpu_map,obj_times,keep
    
    @catch_critical()
    def route_objects(self,obj_ids,priors,device_idxs,camera_idxs,run_device_ids = None):
        """
        Batches input objects onto specified devices
        :param obj_ids - tensor of int object IDs
        :param priors - tensor of size [n_objs,state_size]
        :param device_idxs - int tensor of size [n_objs] indexing GPU on which each obj will be queried
        :param camera_idxs - int tensor of size [n_objs] indexing self.cam_names (global camera ordering)
        :run_device_ids - list of size [n_gpus] with corresponding CUDA device idx for each index. This
                           is to avoid list skip indexing trouble e.g. when CUDA_devices = [0,2,3,4] 
        
        returns - prior_stack - list of size n_devices, where each list element i is:
            (obj_ids,priors,gpu_cam_idx,cam_names)
            obj_ids - subset of input obj_ids on device i
            priors - subset of input priors on device i
            gpu_cam_idx - index of which camera frame from among all camera frames on gpu i
            cam_names - lost of camera names for each object in output obj_ids
        """
        
        # Override
        run_device_ids = None
        
        # if no device ids are specified for this run, assume cuda device ids are contiguous
        if run_device_ids is None:
            run_device_ids = [i for i in range(torch.cuda.device_count())]
            
        
        # no objects
        try:
            if len(obj_ids) == 0:
                return [[[],[],[],[]] for i in range(len(run_device_ids))]
        except:
            return [[[],[],[],[]] for i in range(len(run_device_ids))]

        
        prior_stack = []
        for gpu_id in run_device_ids:
            
            # get all indices into device_idxs where device_idxs[i] == gpu_id
            selected = torch.where(device_idxs == gpu_id,torch.ones(device_idxs.shape),torch.zeros(device_idxs.shape)).nonzero().squeeze(1)
            #selected = torch.where(device_idxs < 50,torch.ones(device_idxs.shape),torch.zeros(device_idxs.shape)).nonzero().squeeze(1)

            
            selected_cams = camera_idxs[selected]
            selected_gpu_cam_idx = torch.tensor([self.cam_gpu_idx[val][1] for val in selected_cams])
            selected_cam_names = [self.cam_names[i] for i in selected_cams]
            
            gpu_priors = (obj_ids[selected],priors[selected,:].to(gpu_id),selected_gpu_cam_idx,selected_cam_names)
            prior_stack.append(gpu_priors)
        
        return prior_stack
    
class HeuristicDeviceMap(DeviceMap):
    
    @catch_critical()
    def __init__(self,camera_list = None,camera_priorities = []):
        super(HeuristicDeviceMap, self).__init__(camera_list = camera_list,camera_priorities=camera_priorities)
        
        # TODO move this to the config
        # add camera priority
        priority_dict = {"c1":1,
                    "c2":100,
                    "c3":1000,
                    "c4":1000,
                    "c5":100,
                    "c6":1,
                    
                    "c01":1,
                    "c02":100,
                    "c03":1000,
                    "c04":1000,
                    "c05":100,
                    "c06":1}
        try:
            self.priority = torch.tensor([priority_dict[re.search("c\d",cam).group(0)] for cam in self.cam_names_extended])
        except KeyError:
            self.priority = torch.tensor([priority_dict[re.search("c\d\d",cam).group(0)] for cam in self.cam_names_extended])
            
    @catch_critical()
    def map_cameras(self,tstate,ts):
        """
        MAPPING:
            constraints:
                object is within camera range
                camera reports timestamp
            preferences:
                interior camers (c3 and c4, then c2 and c5, then c1 and c6)
                camera with center of FOV closest to object
            
        :param - tstate - TrackState object
        :param - ts - list of size [n_cameras] with reported timestamp for each, or torch.nan if ts is invalid
        
        :return list of size n_objs with camera name for each
        """
        
        # TODO - may need to map ts into the correct order of self.cam_ids
        
        # get object positions as tensor
        ids,states = tstate() # [n_objects,state_size]
        
        # store useful dimensions
        #n_c = len(self.cam_devices)
        n_c = len(self.cam_names_extended)
        n_o = len(ids)
        
        if n_o == 0:
            return [],[],[]
        
        ## create is_visible, [n_objects,n_cameras]
        # broadcast both to [n_objs,n_cameras,2]
        states_brx = states[:,0].unsqueeze(1).unsqueeze(2).expand(n_o,n_c,2)
        cams_brx = self.cam_extents[:,[0,1]].unsqueeze(0).expand(n_o,n_c,2)
        # get map of where state is within range
        x_pass = torch.sign(states_brx-cams_brx)
        x_pass = ((torch.mul(x_pass[:,:,0],x_pass[:,:,1]) -1 )* -0.5).int()  # 1 if inside, 0 if outside
        #is_visible = x_pass
        
        #cams_bry = self.cam_extents[:,2].unsqueeze(0).expand(n_o,n_c)
        states_bry = states[:,1].unsqueeze(1).unsqueeze(2).expand(n_o,n_c,2)
        cams_bry = self.cam_extents[:,[2,3]].unsqueeze(0).expand(n_o,n_c,2)
        y_pass = torch.sign(states_bry - cams_bry)
        y_pass = ((torch.mul(y_pass[:,:,0],y_pass[:,:,1]) -1 )* -0.5).int()  # 1 if inside, 0 if outside
        
        is_visible = x_pass * y_pass
        
        
        ## create ts_valid, [n_objs,n_cameras]
        if False: # bypass ts_valid consideration
            object_ts = - tstate.get_dt(0) # time at which each object was left
            object_ts = object_ts.unsqueeze(1).expand(n_o,n_c)
            try:
                frame_ts = torch.tensor(ts).unsqueeze(0).expand(n_o,n_c)
                ts_valid = torch.where(object_ts-frame_ts < 0, torch.ones([n_o,n_c]),torch.zeros([n_o,n_c]))
            except:
                ts_valid = torch.ones([n_o,n_c])
       
        # print("\nts_valid: {}\n".format(torch.mean(ts_valid)))
        #ts_valid = [(0 if item < 1 else 1) for item in ts]
        #ts_valid = torch.tensor(ts_valid).int().unsqueeze(0).expand(n_o,n_c)
        #ts_valid = torch.nan_to_num(ts+1).clamp(0,1).int().unsqueeze(0).expand(n_o,n_c)
        #ts_valid = torch.ones([n_o,n_c])
        
        
        # here we need to expand ts from n_cameras to n_extended_cameras (one per side)
        ts_expanded = torch.tensor(ts,dtype = torch.double)[self.cam_expansion_map]
        # invalid timestamps are -inf, so we just need to make sure that larger than a sufficiently offset start time (say -10 s)
        ts_valid = torch.where(ts_expanded > -10, 1, 0)
        ts_valid = ts_valid.unsqueeze(0).expand(n_o,n_c)
        
        # create priority, [n_objs,n_cameras]
        priority = self.priority.unsqueeze(0).expand(n_o,n_c)
        
        # create distance, [n_objs,n_cameras]        
        center_x = cams_brx.sum(dim = 2)/2.0
        center_y = cams_bry.sum(dim = 2)/2.0
        
        dist = ((center_x - states_brx[:,:,0]).pow(2) + (center_y - states_bry[:,:,0]).pow(2)).sqrt()
        
        score = 1/dist * priority * ts_valid * is_visible
        
        max_scores,cam_map = score.max(dim = 1)
        
        keep = max_scores.nonzero().squeeze()

        # need to squash cam_map so that it is direction-agnostic for getting ts
        cam_map = torch.tensor([self.cam_idxs[self.cam_names_extended[i].split("_")[0]] for i in cam_map])

        obj_times = [ts[idx] for idx in cam_map]
        
        
        #TODO need to unmap cameras to idxs here?
        
        return cam_map,obj_times,keep
    
    def filter_by_extents(self,detections,det_cams):
        """
        returns a list of indices
        """
        try:        
            detection_extents = [torch.tensor(self.cam_extents_dict[cam_name]) for cam_name in det_cams]
        except:
            det_cams = [det_cams[i] + ("_eb" if detections[i,5] == 1 else "_wb") for i in range(len(det_cams))]
            lis = []
            for cam_name in det_cams:
                try:
                   lis.append(torch.tensor(self.cam_extents_dict[cam_name]))
                except KeyError:
                    lis.append(torch.tensor([0,0,0,0]))
            detection_extents = lis
            if len(detection_extents) == 0:
                return torch.empty([0])
        detection_extents = torch.stack(detection_extents)
     
        keep = torch.where(detections[:,0] > detection_extents[:,0], 1,0) *    \
                torch.where(detections[:,0] < detection_extents[:,1], 1,0) *   \
                torch.where(detections[:,1] > detection_extents[:,2], 1,0) *   \
                torch.where(detections[:,1] < detection_extents[:,1], 1,0)
        keep = keep.nonzero().squeeze()
        
        if len(keep.shape) == 0:
            keep = []
        return keep