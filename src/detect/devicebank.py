import torch.multiprocessing as mp
import torch
import copy 

#from ..log_init import logger
from i24_logger.log_writer import logger,catch_critical

from ..util.misc import Timer

class DeviceBank():
    """
    DeviceBank  maintains a list of DeviceHandler objects, one per GPU device.
    Each DeviceHandler object communicates with a separate process to execute Pipelines
    in parallel. The main purpose of the DeviceBank is as a simple input-splitter and output-combiner,
    as well as to initialize each DeviceHandler with the correct parameters
    """
    
    
    def __init__(self,device_ids,lpipelines,device_cam_names,ctx):
        """
        gpu_cam_names = list of lists, with one list per gpu, and camera names assigned to that gpu in the list
        """
        self.device_ids = device_ids
        
        self.send_queues = {}
        self.receive_queues = []
        self.handlers = []
        
        #self.manager = mp.Manager()
        #mp.set_start_method('spawn')
        #ctx = mp.get_context('spawn')
        
        # create handler objects
        idx = 0
        for dev_id in device_ids:
            
            print("Device {} is processing cameras {}".format(dev_id,device_cam_names[dev_id]))
            # store which cameras the pipeline will be processing
            #pipelines = copy.deepcopy(pipelines)
            pipelines = lpipelines[idx]
            idx += 1
            for p in pipelines:
                p.set_cam_names(device_cam_names[dev_id])
                p.set_device(dev_id)
            in_queue = ctx.Queue()
            out_queue = ctx.Manager().Queue() # for some reason this is much faster queue but can't send CUDA tensors recieved from other processes
            handler = ctx.Process(target=handle_device, args=(in_queue,out_queue,pipelines,dev_id,device_cam_names[dev_id]),daemon = True)
            handler.start()
            
            self.handlers.append(handler)
            self.send_queues[dev_id] = in_queue
            self.receive_queues.append(out_queue)
        
        del lpipelines
        
        logger.debug("Main started {} pipeline handler processes".format(len(self.device_ids)))
        self.device_cam_names = device_cam_names
        
        self.ctx = ctx   
        
        self.tm = Timer()
        self.batches_processed = 0
      
    def __del__(self):
        for h in self.handlers:
            h.terminate()
            h.join()
        for sq in self.send_queues:
            del sq
        for rq in self.recieve_queues:
            del rq
            
        
    @catch_critical()
    def __call__(self,prior_stack,frames,pipeline_idx=0):
        """
        :prior_stack - list of items (obj_ids,priors,cam_idx), one for each device
                      where obj_ids - tensor of size [n_objs_on_gpu]
                            priors  - tensor of size [n_objs_on_gpu,state_size]
                            cam_idx - tensor of size [n_obs_on_gpu] with index into cameras on that GPU
        :param frames - list of tensors - [n_cameras_on_gpu,channels,im_height,im_width], one for each device
        :param pipeline_idx - int
        
        
        """
        
        self.batches_processed += 1
        if self.batches_processed % 50 == 0:
            logger.info("DeviceBank Time Util: {}".format(self.tm),extra = self.tm.bins())

        
        # send tasks
        self.tm.split("Send",SYNC = True)
        for i in range(len(prior_stack)):
            if i in self.send_queues.keys():
                # prior stack i will always correspond to device 0
                self.send_queues[i].put((prior_stack[i],frames[i],pipeline_idx))
                #print("Sent {} frames to device {}".format(frames[i].shape[0],i))
        # get results
        result = []
        for i in range(len(self.receive_queues)):
            self.tm.split("Recieve {}".format(i),SYNC = True)
            result.append(self.receive(i))
        
        # concat and return results
        self.tm.split("Concat",SYNC = True)
        lens = [len(item[0]) for item in result]
        #print(lens)
        result = self.concat_stack(result)    
        
        self.tm.split("Waiting")
        return result
    
    
    
    
    def concat_stack(self,inp):
        """
        Recieves a list of n lists of m elements each, and returns a list of m results
        where each result is a concatenated tensor (if type = tensor) or else a concatenated list
        
        :param inp - list of lists of length n and m respectively
        :return out_stack - list m of tensors or lists
        """
        
        stack = [[] for _ in range(len(inp[0]))]
        
        for item in inp:
            [stack[i].append(item[i]) for i in range(len(item))]
        
        out_stack= []
        for out in stack:
            if type(out[0]) == torch.Tensor:
                out = torch.cat(out,dim = 0)
            elif type(out[0]) == list:
                out = [item for sublist in out for item in sublist]

                
            out_stack.append(out)
        return out_stack
        
        
    def receive(self,idx):
        while True:
            if self.receive_queues[idx].empty():
                continue 
            else:                
                result = self.receive_queues[idx].get() 
                return result
                
    
    def __del__(self):
        for pid in self.handlers:
            pid.terminate()
            pid.join()
            
        # for q in self.send_queues:
        #     del q
        # for q in self.receive_queues:
        #     del q
        
        # del self.ctx 
        # del self.manager
            
            
        # TODO log DeviceBank shutdown
        
def handle_device(in_queue,out_queue,pipelines,device_id,this_dev_cam_names):
   
    # intialize
    #device = torch.cuda.device("cuda:{}".format(device_id) if device_id != -1 else "cpu")
    torch.cuda.set_device(device_id)
    [pipelines[i].set_device(device_id) for i in range(len(pipelines))]
    
    #from i24_logger.log_writer import logger
    logger.set_name("Tracking Device Handler {}".format(device_id))
    logger.debug("Device handler {} initialized to process {}".format(device_id,this_dev_cam_names))
    
    while True:
        
        if in_queue.empty():
            continue 
        else:
            # try to get set of inputs from in_queue
            # each item on this queue is (prior_stack,frames,pipeline_idx)
            inputs = in_queue.get(timeout = 0) 
    
        if inputs is None:
            break
            #TODO log device handler shutdown
        
        (prior_data,frames,pipeline_idx) = inputs
        torch.cuda.empty_cache()
        result = pipelines[pipeline_idx](frames,priors = prior_data)
        out_queue.put(result)
        