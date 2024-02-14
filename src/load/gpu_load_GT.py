import torch
import PyNvCodec as nvc
import PytorchNvCodec as pnvc

import queue

import configparser
import torch

import os
import numpy as np
import time
import re

import cv2
from PIL import Image

import _pickle as pickle

from torchvision.transforms import functional as F


class MCLoader():
    """
    Loads multiple video files in parallel with no timestamp parsing. Timestamps are instead parsed from an associated file
    
    """
    
    def __init__(self,directory,timestamp_file,mapping_file,ctx,resize = (1920,1080)):
        
        cp = configparser.ConfigParser()
        cp.read(mapping_file)
        mapping = dict(cp["DEFAULT"])
       
        for key in mapping.keys():
            parsed_val = int(mapping[key])
            mapping[key] = parsed_val
    
        self.cam_devices = mapping

        cam_sequences = {}        
        for file in os.listdir(directory):
            sequence = os.path.join(directory,file)
            cam_name = re.search("p\dc\d",sequence).group(0)
            
            if cam_name in self.cam_devices.keys():
                cam_sequences[cam_name] = sequence
        
        self.dummy_ts = 0
        
        
        # device loader is a list of lists, with list i containing all loaders for device i (hopefully in order but not well enforced by dictionary so IDK)
        self.device_loaders = [[] for i in range(torch.cuda.device_count())]
        self.cam_names = [[] for i in range(torch.cuda.device_count())]
        for key in cam_sequences.keys():
            dev_id = self.cam_devices[key]
            sequence = cam_sequences[key]
            loader = GPUBackendFrameGetter(sequence,dev_id,ctx,resize = resize)
            
            self.device_loaders[dev_id].append(loader)
            self.cam_names[dev_id].append(key)
        
        # parse timestamp_file
        with open(timestamp_file,"rb") as f:
           _,timestamps,_ = pickle.load(f)
        
        self.timestamps = timestamps # dict
        self.timestamp_idx = dict([(key,-1) for key in timestamps[0].keys()])
        self.start_timestamp = max([self.timestamps[0][key] for key in self.timestamps[0]]) # get max of all camera timestamps
        
            
    def get_frames(self,target_time= None,tolerance = 1/60.0):
        
        try:
            if target_time is None:
                # accumulators
                frames = [[] for i in range(torch.cuda.device_count())]
                timestamps = []
                for dev_idx,this_dev_loaders in enumerate(self.device_loaders):
                    for l_idx,loader in enumerate(this_dev_loaders):
                        frame = next(loader)
                        frames[dev_idx].append(frame)
                        
                        # TODO - we need to correctly index self.timestamps but I forget if it's a list or dict
                        cname = self.cam_names[dev_idx][l_idx]
                        self.timestamp_idx[cname] += 1
                        ts = self.timestamps[self.timestamp_idx[cname]][cname]
                        timestamps.append(ts)
            else:
                # accumulators
                frames = [[] for i in range(torch.cuda.device_count())]
                timestamps = []
                
                # advance each camera loader
                for dev_idx,this_dev_loaders in enumerate(self.device_loaders):
                    for l_idx,loader in enumerate(this_dev_loaders):
                        cname = self.cam_names[dev_idx][l_idx]
    
                        
                        # require an advancement of at least one frame
                        frame = next(loader)
                        self.timestamp_idx[cname] += 1
                        ts = self.timestamps[self.timestamp_idx[cname]][cname]
                        
                        # skip frames as necessary to sync times
                        while ts + tolerance < target_time:
                            frame= next(loader)
                            self.timestamp_idx[cname] += 1
                            ts = self.timestamps[self.timestamp_idx[cname]][cname]
                        
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
            
            return out,timestamps
    
        except: # end of input
            return None, None
    
class GPUBackendFrameGetter:
    def __init__(self,file,device,ctx,buffer_size = 20,resize = (1920,1080)):
        
        # create shared queue
        self.queue = ctx.Queue()
        self.frame_idx = -1
        self.device = device  
        
        self.file = file
        self.worker = ctx.Process(target=load_queue_vpf, args=(self.queue,file,device,buffer_size,resize),daemon = True)
        self.worker.start()   
            

    def __len__(self):
        """
        Description
        -----------
        Returns number of frames in the track directory
        """
        
        return 1000000
    
    
    def __next__(self):
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
            frame = self.queue.get(timeout = 10)
            return frame
            
        except queue.Empty: #TODO - implement shutdown
            self.worker.terminate()
            self.worker.join()
            return None
        
        
def load_queue_vpf(q,file,device,buffer_size,resize):
    resize = (resize[1],resize[0])
    gpuID = device
    device = torch.cuda.device("cuda:{}".format(gpuID))
    
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
    
    while True:
        if q.qsize() < buffer_size:
            pkt = nvc.PacketData()
            
            # Obtain NV12 decoded surface from decoder;
            rawSurface = nvDec.DecodeSingleSurface(pkt)
            if (rawSurface.Empty()):
                break
            
            # Convert to RGB interleaved;
            rgb_byte = to_rgb.Execute(rawSurface, cc_ctx)
        
            # Convert to RGB planar because that's what to_tensor + normalize are doing;
            rgb_planar = to_planar.Execute(rgb_byte, cc_ctx)
        
            # Create torch tensor from it and reshape because
            # pnvc.makefromDevicePtrUint8 creates just a chunk of CUDA memory
            # and then copies data from plane pointer to allocated chunk;
            surfPlane = rgb_planar.PlanePtr()
            surface_tensor = pnvc.makefromDevicePtrUint8(surfPlane.GpuMem(), surfPlane.Width(), surfPlane.Height(), surfPlane.Pitch(), surfPlane.ElemSize())
            surface_tensor.resize_(3, target_h,target_w)
            
            try:
                surface_tensor = torch.nn.functional.interpolate(surface_tensor.unsqueeze(0),resize).squeeze(0)
            except:
                raise Exception("Surface tensor shape:{} --- resize shape: {}".format(surface_tensor.shape,resize))
        
            # This is optional and depends on what you NN expects to take as input
            # Normalize to range desired by NN. Originally it's 
            surface_tensor = surface_tensor.type(dtype=torch.cuda.FloatTensor)/255.0
            
            
            # apply normalization
            surface_tensor = F.normalize(surface_tensor,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            # TODO - I think the model is trained with inverted RGB channels unfornately so this is a temporary fix
            #surface_tensor = surface_tensor[[2,1,0],:,:] 

            
            frame = surface_tensor
            q.put(frame)
            
            
if __name__ == "__main__":
    import torch.multiprocessing as mp
    ctx = mp.get_context('spawn')
    test = MCLoader("/home/derek/Data/dataset_beta/sequence_0","/home/derek/Documents/derek/3D-playground/linear_spacing_splines_0.cpkl","/home/derek/Documents/i24/i24_track/config/lambda_cerulean/cam_devices.config",ctx)
    
    test.get_frames(test.start_timestamp)
    
    count = 1
    while True:
        count += 1
        test.get_frames()
        print("On frame {}".format(count))