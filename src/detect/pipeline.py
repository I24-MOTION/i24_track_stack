import torch
from torchvision.ops import roi_align
from i24_configparse import parse_cfg
from ..util.bbox import im_nms,space_nms,state_nms

# imports for specific detectors
from .retinanet_3D.retinanet.model             import resnet50 as Retinanet3D
from .retinanet_3D_emb.retinanet.model         import resnet50 as Retinanet3D_emb
from .retinanet_3D_multi_frame.retinanet.model import resnet34 as Retinanet3D_multi_frame


from i24_logger.log_writer import logger,catch_critical

from ..util.misc import Timer

@catch_critical()
def get_Pipeline(name,hg):
    """
    getter function that takes a string (class name) input and returns an instance
    of the named class
    """
    if name == "RetinanetFullFramePipeline":
        pipeline = RetinanetFullFramePipeline(hg)
    elif name == "RetinanetCropFramePipeline":
        pipeline = RetinanetCropFramePipeline(hg)
    elif name == "RetinanetFullFramePipelineEmb":
        pipeline = RetinanetFullFramePipelineEmb(hg)
    elif name == "RetinanetFullFramePipelineMulti":
        pipeline = RetinanetFullFramePipelineMulti(hg)
    elif name == "RetinanetCropFramePipelineMulti":
        pipeline = RetinanetCropFramePipelineMulti(hg)
    else:
        raise NotImplementedError("No DetectPipeline child class named {}".format(name))
    
    return pipeline
    

class DetectPipeline():
    """
    The DetectPipeline provides a wrapper class for all operations carried out on a single
    GPU for a single frame; generally, this includes prepping frames for detection,
    performing inference with self.detector, and any subsequent post-detection steps
    such as low-confidence filtering or best detection selection based on priors
    """
    
    def __init__(self):
        raise NotImplementedError
        
    def prep_frames(self,frames,priors = None):
        """
        Receives a stack of frames as input and returns a stack of frames as output.
        All frames in each stack are expected to be on the same device as the DetectPipeline object. 
        In the default case, no changes are made. Note that if priors are used, they should 
        be converted to im pixels here
        param: frames (tensor of shape [B,3,H,W])
        return: frames (tensor of shape [B,3,H,W])
        """
        
        return frames
        
    def detect(self,frames):
        with torch.no_grad():
            result = self.detector(frames)
        return result
        
    def post_detect(self,detection_result,priors = None):
        
        """
        At a minimum this function should apply hg to convert to state/space
        """
        return detection_result
    
    @catch_critical()
    def __call__(self,frames,priors):
        
        [ids,priors,frame_idx,cam_names] = priors
        
        # no frames assigned to this GPU
        if frames.shape[0] == 0:
            logger.warning("No frames assigned to device: {}".format(self.device))
            del frames
            return torch.empty([0,6]) , torch.empty(0), torch.empty(0), [], []
        
        prepped_frames = self.prep_frames(frames,priors = priors)
        detection_result = self.detect(prepped_frames)
        #print("Device {} detection result length: ".format(self.device),len(detection_result))
        detections,confs,classes,detection_cam_names  = self.post_detect(detection_result,priors = priors)
        
        # Associate
        #matchings = self.associate(ids,priors,detections,self.hg)
        matchings = []

        del frames,priors
        return detections,confs,classes,detection_cam_names,matchings
    
    def set_cam_names(self,cam_names):
        self.cam_names = cam_names
    
    @catch_critical()
    def set_device(self,device_id = -1):
        # configure detector
        self.device = torch.device("cuda:{}".format(device_id) if device_id != -1 else "cpu")
        
        self.detector = self.detector.to(self.device)
        self.detector.eval()
        self.detector.training = False
        
        if device_id != -1:
            torch.cuda.set_device(device_id)
    
    
    
class RetinanetFullFramePipeline(DetectPipeline):
    """
    DetectPipeline that applies Retinanet Detector on full object frames
    """
    
    def __init__(self,hg):
        
        # load configuration file parameters
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)

        
        # initialize detector
        self.detector = Retinanet3D(self.n_classes)
    
        # quantize model         
        if self.quantize:
            self.detector = self.detector.half()
        
        # load detector weights
        self.detector.load_state_dict(torch.load(self.weights_file))
    
        self.hg = hg # store homography

        self.cam_names = None
        
        
    def detect(self,frames):
        
        #logger.debug("Log message test from pipeline: {}".format(self.device))
        
        with torch.no_grad():
            result = self.detector(frames,MULTI_FRAME = True)
        return result
    
    def post_detect(self,detection_result,priors = None):
        confs,classes,detections,detection_idxs = detection_result # detection_idx = which frame from idx each detection is from
        #confs,classes = torch.max(classes, dim = 2) 
        detection_cam_names = [] # dummy value in case no objects returned
        
        
        # reshape detections to form [d,8,2] 
        detections = detections.reshape(-1,10,2)
        detections = detections[:,:8,:] # drop 2D boxes
        
        
        # low confidence filter
        if len(confs) > 0:
            mask           = torch.where(confs > self.min_conf,torch.ones(confs.shape,device = confs.device),torch.zeros(confs.shape,device = confs.device)).nonzero().squeeze(1)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask,:,:]
            detection_idxs = detection_idxs[mask]
        
        # im space NMS 
        if len(confs) > 0 and self.im_nms_iou < 1:
            mask           = im_nms(detections,confs,threshold = self.im_nms_iou,groups = detection_idxs)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask,:,:]
            detection_idxs = detection_idxs[mask]
        
        if len(confs) > 0:
            detection_cam_names = [self.cam_names[i] for i in detection_idxs]
            
            # Use the guess and refine method to get box heights
            detections = self.hg.im_to_state(detections,name = detection_cam_names,classes = classes)
            detections = self.hg.state_to_space(detections)
            # state space NMS
            mask           = space_nms(detections,confs,threshold = self.space_nms_iou)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask,:,:]
            detection_idxs = detection_idxs[mask]
            detection_cam_names = [self.cam_names[i] for i in detection_idxs]
            
            if len(confs) > 0:
                detections = self.hg.space_to_state(detections)
        
        # finally, move back to cpu
        detections = detections.cpu()
        confs = confs.cpu()
        classes = classes.cpu()
                
        return [detections,confs,classes,detection_cam_names]


class RetinanetFullFramePipelineMulti(DetectPipeline):
    """
    DetectPipeline that applies Retinanet Detector on full object frames
    """
    
    def __init__(self,hg):
        
        # load configuration file parameters
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)

        
        # initialize detector
        self.detector = Retinanet3D_multi_frame(self.n_classes)
        
        conv1 = torch.nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.weight.data[:,:3,:,:] = self.detector.conv1.weight.data.clone()
        conv1.weight.data[:,3:,:,:] = self.detector.conv1.weight.data.clone()
        self.detector.conv1 = conv1
    
        
        # load detector weights
        self.detector.load_state_dict(torch.load(self.weights_file))
        
        self.detector.training = False
        self.detector.eval()
    
        # quantize model         
        if self.quantize:
            self.detector = self.detector.half()
            for layer in self.detector.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.float()
    
        self.hg = hg # store homography

        self.cam_names = None
            
        self.prev_frame = None
        
        self.tm = Timer()
        self.batches_processed = 0
        
    def detect(self,frames):
        
        self.batches_processed += 1
        if self.batches_processed %50 == 0:
            logger.info("Full Frame Pipeline Time Util: {}".format(self.tm),extra = self.tm.bins())
            
        self.tm.split("half frames")

        #logger.debug("Log message test from pipeline: {}".format(self.device))
        self.detector.training = False
        self.detector.eval()

        if self.prev_frame is None:
            self.prev_frame = frames.clone()            
        
        frames_cat = torch.cat((frames,self.prev_frame),dim = 1)
        
        self.prev_frame = 0.9* self.prev_frame + 0.1*frames.clone()
        
        if self.quantize:
            frames_cat = frames_cat.half()
        
        self.tm.split("detect")
        with torch.no_grad():
            result = self.detector(frames_cat,MULTI_FRAME = True)
        return result
    
    def post_detect(self,detection_result,priors = None):
        self.tm.split("post")
        confs,classes,detections,detection_idxs,embeddings = detection_result # detection_idx = which frame from idx each detection is from
        #confs,classes = torch.max(classes, dim = 2) 
        detection_cam_names = [] # dummy value in case no objects returned
        
        #print("In detector length: {}".format(len(detections)))
        
        # reshape detections to form [d,8,2] 
        detections = detections.reshape(-1,10,2)
        detections = detections[:,:8,:] # drop 2D boxes
        
        
        # low confidence filter
        if len(confs) > 0:
            mask           = torch.where(confs > self.min_conf,torch.ones(confs.shape,device = confs.device),torch.zeros(confs.shape,device = confs.device)).nonzero().squeeze(1)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask,:,:]
            detection_idxs = detection_idxs[mask]
        
        # im space NMS 
        if len(confs) > 0 and self.im_nms_iou < 1:
            mask           = im_nms(detections,confs,threshold = self.im_nms_iou,groups = detection_idxs)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask,:,:]
            detection_idxs = detection_idxs[mask]
        
        if len(confs) > 0:
            detection_cam_names = [self.cam_names[i] for i in detection_idxs]
            
            # Use the guess and refine method to get box heights
            detections = self.hg.im_to_state(detections,name = detection_cam_names,classes = classes)
            #detections = self.hg.state_to_space(detections)
            # state space NMS
            mask           = state_nms(detections,confs,threshold = self.space_nms_iou)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask,:]
            detection_idxs = detection_idxs[mask]
            detection_cam_names = [self.cam_names[i] for i in detection_idxs]
        
        self.tm.split("waiting")
        if len(confs) > 0:
            #detections = self.hg.space_to_state(detections)
    
            # finally, move back to cpu
            detections = detections.cpu()
            confs = confs.cpu()
            classes = classes.cpu()
                    
            return [detections,confs,classes,detection_cam_names]   
    
        else:
            return torch.empty([0,6]) , torch.empty(0), torch.empty(0), []

    
class RetinanetFullFramePipelineEmb(DetectPipeline):
    """
    DetectPipeline that applies Retinanet Detector on full object frames
    """
    
    def __init__(self,hg):
        
        # load configuration file parameters
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)

        
        # initialize detector
        self.detector = Retinanet3D_emb(self.n_classes)
    
        # quantize model         
        if self.quantize:
            self.detector = self.detector.half()
        
        # load detector weights
        self.detector.load_state_dict(torch.load(self.weights_file))
    
        self.hg = hg # store homography

        self.cam_names = None
        
        
    def detect(self,frames):
        
        #logger.debug("Log message test from pipeline: {}".format(self.device))
        
        with torch.no_grad():
            result = self.detector(frames,MULTI_FRAME = True)
        return result
    
    def post_detect(self,detection_result,priors = None):
        confs,classes,detections,detection_idxs,embeddings = detection_result # detection_idx = which frame from idx each detection is from
        #confs,classes = torch.max(classes, dim = 2) 
        detection_cam_names = [] # dummy value in case no objects returned
        
        
        # reshape detections to form [d,8,2] 
        detections = detections.reshape(-1,10,2)
        detections = detections[:,:8,:] # drop 2D boxes
        
        
        # low confidence filter
        if len(confs) > 0:
            mask           = torch.where(confs > self.min_conf,torch.ones(confs.shape,device = confs.device),torch.zeros(confs.shape,device = confs.device)).nonzero().squeeze(1)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask,:,:]
            detection_idxs = detection_idxs[mask]
        
        # im space NMS 
        if len(confs) > 0 and self.im_nms_iou < 1:
            mask           = im_nms(detections,confs,threshold = self.im_nms_iou,groups = detection_idxs)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask,:,:]
            detection_idxs = detection_idxs[mask]
        
        if len(confs) > 0:
            detection_cam_names = [self.cam_names[i] for i in detection_idxs]
            
            # Use the guess and refine method to get box heights
            detections = self.hg.im_to_state(detections,name = detection_cam_names,classes = classes)
            #detections = self.hg.state_to_space(detections)
            # state space NMS
            mask           = state_nms(detections,confs,threshold = self.space_nms_iou)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask,:,:]
            detection_idxs = detection_idxs[mask]
            detection_cam_names = [self.cam_names[i] for i in detection_idxs]
            
            if len(confs) > 0:
                detections = self.hg.space_to_state(detections)
        
        # finally, move back to cpu
        detections = detections.cpu()
        confs = confs.cpu()
        classes = classes.cpu()
                
        return [detections,confs,classes,detection_cam_names]   
    
    
        
class RetinanetCropFramePipeline(DetectPipeline):
        
    def __init__(self,hg,):
        #  load configuration file parameters
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)

        
        # initialize detector
        self.detector = Retinanet3D(self.n_classes)
    
        self.detector.eval()
        self.detector.training = False
        
        # quantize model         
        if self.quantize:
            self.detector = self.detector.half()
        
        # load detector weights
        self.detector.load_state_dict(torch.load(self.weights_file))
    
        self.hg = hg # store homography

        self.cam_names = None
        
        self.tm = Timer()
        
        self.batches_processed = 0
        
    @catch_critical()
    def __call__(self,frames,priors):
        
        #[ids,priors,frame_idx,cam_names] = priors
        
        # periodically log time utilization
        self.batches_processed += 1
        if self.batches_processed %500 == 0:
            logger.info("Crop Pipeline Time Util: {}".format(self.tm),extra = self.tm.bins())
            
            
        # no frames assigned to this GPU
        if frames.shape[0] == 0 or len(priors[0]) == 0:
            del frames
            return torch.empty([0,6]) , torch.empty(0), torch.empty(0), [], torch.empty(0)
        
        self.tm.split("Pre",SYNC=True)
        prepped_frames,crop_boxes = self.prep_frames(frames,priors = priors)
        self.tm.split("Detect",SYNC=True)
        detection_result = self.detect(prepped_frames)

        detections,classes,confs,detection_cam_names,matchings  = self.post_detect(detection_result,priors,crop_boxes)
        
        self.tm.split("GPU->CPU",SYNC = True)
        del frames,priors,crop_boxes
        ret =  detections.cpu(),confs.cpu(),classes.cpu(),detection_cam_names,matchings
        
        self.tm.split("Waiting",SYNC = True)
        return ret
        
        
    
    def prep_frames(self,frames,priors):
        """
        priors - list of [ids,priors (nx6 tensor), frame_idx in stack, cam_name for prior]
        frames - 
        """

        self.tm.split("Pre:get",SYNC=True)
        ids,objs,frame_idxs,cam_names = priors
        
        # 1. State to im
        self.tm.split("Pre:state->im",SYNC=True)
        objs_im = self.hg.state_to_im(objs,name = cam_names)
    
        # 2. Get expanded boxes
        self.tm.split("Pre:compute_crops",SYNC=True)
        crop_boxes = self._get_crop_boxes(objs_im)
        
        # 3. Crop
        self.tm.split("Pre:crop",SYNC=True)
        cidx = frame_idxs.unsqueeze(1).to(self.device).double()
        torch_boxes = torch.cat((cidx,crop_boxes),dim = 1)
        crops = roi_align(frames,torch_boxes.float(),(self.crop_size,self.crop_size))
        
        return crops,crop_boxes
        
        
    def detect(self,frames):
        with torch.no_grad():
            result = self.detector(frames,LOCALIZE = True)
        return result
    
    def post_detect(self,detection_result,priors,crop_boxes):
        
        self.tm.split("Post:->GPU",SYNC=True)
        ids,objs,frame_idxs,cam_names = priors
        objs = objs.to(self.device)
    
        self.tm.split("Post:classes",SYNC=True)        
        reg_boxes, classes = detection_result
        confs,classes = torch.max(classes, dim = 2)

        # 2. local to global mapping
        self.tm.split("Post:local->global",SYNC=True)        
        reg_boxes = self._local_to_global(reg_boxes,crop_boxes)    

        # 1. Keep top k boxes
        self.tm.split("Post:topk",SYNC=True)  
        top_idxs = torch.topk(confs,self.keep_boxes,dim = 1)[1]
        row_idxs = torch.arange(reg_boxes.shape[0]).unsqueeze(1).repeat(1,top_idxs.shape[1])
        
        reg_boxes = reg_boxes[row_idxs,top_idxs,:,:]
        confs =  confs[row_idxs,top_idxs]
        classes = classes[row_idxs,top_idxs] 
        
        
        # 3. Convert to space    #### THIS IS THE SLOW ONE!!!
        self.tm.split("Post:->space",SYNC=True)  
        n_objs = reg_boxes.shape[0]
        cam_names_repeated = [cam for cam in cam_names for i in range(reg_boxes.shape[1])]
        reg_boxes_state = self.hg.im_to_state(reg_boxes.view(-1,8,2),name = cam_names_repeated,classes = classes.view(-1))
        
 
        # 4. Select best box
        self.tm.split("select_best",SYNC=True)  
        detections, classes, confs = self._select_best_box(objs,reg_boxes_state,confs,classes,n_objs)
        
        
        # note that cam_names and ids directly become detection cameras and detection ids - implicit matching        
        return detections,classes,confs,cam_names,ids
        
        
    def _get_crop_boxes(self,objects):
            """
            Given a set of objects, returns boxes to crop them from the frame
            objects - [n,8,2] array of x,y, corner coordinates for 3D bounding boxes
            
            returns [n,4] array of xmin,xmax,ymin,ymax for cropping each object
            """
            
            # find xmin,xmax,ymin, and ymax for 3D box points
            self.tm.split("Crop:minmax",SYNC=True)
            minx = torch.min(objects[:,:,0],dim = 1)[0]
            miny = torch.min(objects[:,:,1],dim = 1)[0]
            maxx = torch.max(objects[:,:,0],dim = 1)[0]
            maxy = torch.max(objects[:,:,1],dim = 1)[0]
            
            self.tm.split("Crop:scale",SYNC=True)
            w = maxx - minx
            h = maxy - miny
            scale = torch.max(torch.stack([w,h]),dim = 0)[0] * self.box_expansion_ratio
            
            self.tm.split("Crop:xysr",SYNC=True)
            # find a tight box around each object in xysr formulation
            minx2 = (minx+maxx)/2.0 - scale/2.0
            maxx2 = (minx+maxx)/2.0 + scale/2.0
            miny2 = (miny+maxy)/2.0 - scale/2.0
            maxy2 = (miny+maxy)/2.0 + scale/2.0
            
            self.tm.split("Crop:stack",SYNC=True)
            crop_boxes = torch.stack([minx2,miny2,maxx2,maxy2]).transpose(0,1).to(self.device)
            return crop_boxes
        
    def _local_to_global(self,preds,crop_boxes):
        """
        Convert from crop coordinates to frame coordinates
        preds - [n,d,20] array where n indexes object and d indexes detections for that object
        crops_boxes - [n,4] array
        """
        n = preds.shape[0]
        d = preds.shape[1]
        preds = preds.reshape(n,d,10,2)
        preds = preds[:,:,:8,:] # drop 2D boxes
        
        scales = torch.max(torch.stack([crop_boxes[:,2] - crop_boxes[:,0],crop_boxes[:,3] - crop_boxes[:,1]]),dim = 0)[0]
        
        # preds is [n,d,8,2] - expand scale, currently [n], to match
        scales = scales.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,d,8,2)
        
        # scale each box by the box scale / crop size self.cs
        preds = preds * scales / self.crop_size
    
        # shift based on crop box corner
        preds[:,:,:,0] += crop_boxes[:,0].unsqueeze(1).unsqueeze(1).repeat(1,d,8)
        preds[:,:,:,1] += crop_boxes[:,1].unsqueeze(1).unsqueeze(1).repeat(1,d,8)
        
        return preds
    
    def _select_best_box(self,a_priori,preds,confs,classes,n_objs):
        """
        a_priori - [n,6] array of state formulation object priors
        preds    - [n,d,6] array where n indexes object and d indexes detections for that object, in state formulation
        confs   - [n,d] array of confidence for each pred
        classes   - [n,d] array of class prediction for each pred
        returns  - [n,6] array of best matched objects
        """

        
        # convert  preds into space 
        preds_space = self.hg.state_to_space(preds.clone())
        preds_space = preds_space.reshape(n_objs,-1,8,3)
        preds = preds.reshape(n_objs,-1,6)
        
        n = preds_space.shape[0] 
        d = preds_space.shape[1]
        
        # convert into xmin ymin xmax ymax form        
        boxes_new = torch.zeros([n,d,4],device = preds.device)
        boxes_new[:,:,0] = torch.min(preds_space[:,:,0:4,0],dim = 2)[0]
        boxes_new[:,:,2] = torch.max(preds_space[:,:,0:4,0],dim = 2)[0]
        boxes_new[:,:,1] = torch.min(preds_space[:,:,0:4,1],dim = 2)[0]
        boxes_new[:,:,3] = torch.max(preds_space[:,:,0:4,1],dim = 2)[0]
        preds_space = boxes_new
        
        # convert a_priori into space
        a_priori = self.hg.state_to_space(a_priori.clone())
        boxes_new = torch.zeros([n,4],device =  a_priori.device)
        boxes_new[:,0] = torch.min(a_priori[:,0:4,0],dim = 1)[0]
        boxes_new[:,2] = torch.max(a_priori[:,0:4,0],dim = 1)[0]
        boxes_new[:,1] = torch.min(a_priori[:,0:4,1],dim = 1)[0]
        boxes_new[:,3] = torch.max(a_priori[:,0:4,1],dim = 1)[0]
        a_priori = boxes_new
        
        # a_priori is now [n,4] need to repeat by [d]
        a_priori = a_priori.unsqueeze(1).repeat(1,d,1)
        
        # calculate iou for each
        ious = self._md_iou(preds_space.double(),a_priori.double())
        
        # compute score for each box [n,d]
        scores = (1-self.w) * ious + self.w*confs
        
        keep = torch.argmax(scores,dim = 1)
        
        idx = torch.arange(n)
        best_boxes = preds[idx,keep,:]
        cls_preds = classes[idx,keep]
        confs = confs[idx,keep]
        
        
        # gather max score boxes
        
        return best_boxes, cls_preds, confs
    
    def _md_iou(self,a,b):
        """
        a,b - [batch_size ,num_anchors, 4]
        """
        
        area_a = (a[:,:,2]-a[:,:,0]) * (a[:,:,3]-a[:,:,1])
        area_b = (b[:,:,2]-b[:,:,0]) * (b[:,:,3]-b[:,:,1])
        
        minx = torch.max(a[:,:,0], b[:,:,0])
        maxx = torch.min(a[:,:,2], b[:,:,2])
        miny = torch.max(a[:,:,1], b[:,:,1])
        maxy = torch.min(a[:,:,3], b[:,:,3])
        zeros = torch.zeros(minx.shape,dtype=float,device = a.device)
        
        intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
        union = area_a + area_b - intersection
        iou = torch.div(intersection,union)
        
        #print("MD iou: {}".format(iou.max(dim = 1)[0].mean()))
        return iou
            
    
class RetinanetCropFramePipelineMulti(DetectPipeline):
        
    def __init__(self,hg,):
        #  load configuration file parameters
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)

    
        # initialize detector
        self.detector = Retinanet3D_multi_frame(self.n_classes)
        
        conv1 = torch.nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.weight.data[:,:3,:,:] = self.detector.conv1.weight.data.clone()
        conv1.weight.data[:,3:,:,:] = self.detector.conv1.weight.data.clone()
        self.detector.conv1 = conv1
    
        # quantize model         
        if self.quantize:
            self.detector = self.detector.half()
            for layer in self.detector.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.float()
        
        # load detector weights
        self.detector.load_state_dict(torch.load(self.weights_file))
    
        self.hg = hg # store homography

        self.cam_names = None
        
        self.tm = Timer()
        
        self.batches_processed = 0
        
        self.prev_frame = None
        
    @catch_critical()
    def __call__(self,frames,priors):
        
        #[ids,priors,frame_idx,cam_names] = priors
        
        # periodically log time utilization
        self.batches_processed += 1
        if self.batches_processed %500 == 0:
            logger.info("Crop Pipeline Time Util: {}".format(self.tm),extra = self.tm.bins())
            
            
        # no frames assigned to this GPU
        if frames.shape[0] == 0 or len(priors[0]) == 0:
            del frames
            return torch.empty([0,6]) , torch.empty(0), torch.empty(0), [], torch.empty(0)
        
        if self.prev_frame is None:
            self.prev_frame = frames.clone()
        
        frames_cat = torch.cat((frames,self.prev_frame),dim = 1)
        
        
        self.prev_frame = frames.clone()#0.9* self.prev_frame + 0.1*frames.clone()
        
        self.tm.split("Pre",SYNC=True)
        prepped_frames,crop_boxes = self.prep_frames(frames_cat,priors = priors)
        self.tm.split("Detect",SYNC=True)
        
        if self.quantize:
            prepped_frames = prepped_frames.half()
            
        detection_result = self.detect(prepped_frames)

        detections,classes,confs,detection_cam_names,matchings  = self.post_detect(detection_result,priors,crop_boxes)
        
        self.tm.split("GPU->CPU",SYNC = True)
        del frames,priors,crop_boxes
        ret =  detections.cpu(),confs.cpu(),classes.cpu(),detection_cam_names,matchings
        
        self.tm.split("Waiting",SYNC = True)
        return ret
        
        
    
    def prep_frames(self,frames,priors):
        """
        priors - list of [ids,priors (nx6 tensor), frame_idx in stack, cam_name for prior]
        frames - 
        """

        self.tm.split("Pre:get",SYNC=True)
        ids,objs,frame_idxs,cam_names = priors
        
        # 1. State to im
        self.tm.split("Pre:state->im",SYNC=True)
        objs_im = self.hg.state_to_im(objs,name = cam_names)
    
        # 2. Get expanded boxes
        self.tm.split("Pre:compute_crops",SYNC=True)
        crop_boxes = self._get_crop_boxes(objs_im)
        
        #2b. Select only crops entirely within frame
        # mask = (crop_boxes[:,2] > 0).int() * (crop_boxes[:,0] < frames.shape[3]).int() * (crop_boxes[:,3] > 0).int() * (crop_boxes[:,1] < frames.shape[2]).int()
        # mask = mask.nonzero().squeeze()
        
        # if len(mask) == len(ids):
        #     crop_priors = priors
        # else:
        #     crop_boxes = crop_boxes[mask,:]
        #     crop_priors = []
        #     for lis in priors:
        #         new_lis = [lis[idx] for idx in mask]
        #         try:
        #             new_lis = torch.stack(new_lis)
        #         except:
        #             pass
        #         crop_priors.append(lis)
                
        #     frame_idxs = frame_idxs[mask]
        
        # 3. Crop
        self.tm.split("Pre:crop",SYNC=True)
        cidx = frame_idxs.unsqueeze(1).to(self.device).double()
        torch_boxes = torch.cat((cidx,crop_boxes),dim = 1)
        crops = roi_align(frames,torch_boxes.float(),(self.crop_size,self.crop_size))
        
        return crops,crop_boxes
        
        
    def detect(self,frames):
        with torch.no_grad():
            result = self.detector(frames,LOCALIZE = True)
        return result
    
    def post_detect(self,detection_result,priors,crop_boxes):
        
        self.tm.split("Post:->GPU",SYNC=True)
        ids,objs,frame_idxs,cam_names = priors
        objs = objs.to(self.device)
    
        self.tm.split("Post:classes",SYNC=True)        
        reg_boxes, classes, embeddings = detection_result
        confs,classes = torch.max(classes, dim = 2)

        reg_boxes = reg_boxes.float()
        confs = confs.float()

        # 2. local to global mapping
        self.tm.split("Post:local->global",SYNC=True)        
        reg_boxes = self._local_to_global(reg_boxes,crop_boxes)    

        # 1. Keep top k boxes
        self.tm.split("Post:topk",SYNC=True)  
        top_idxs = torch.topk(confs,self.keep_boxes,dim = 1)[1]
        row_idxs = torch.arange(reg_boxes.shape[0]).unsqueeze(1).repeat(1,top_idxs.shape[1])
        
        reg_boxes = reg_boxes[row_idxs,top_idxs,:,:]
        confs =  confs[row_idxs,top_idxs]
        classes = classes[row_idxs,top_idxs] 
        
        
        # 3. Convert to space    #### THIS IS THE SLOW ONE!!!  ## Note taht we could convert base only for the selection and then do second pass for selected ones only (about 1/2 speedup)
        self.tm.split("Post:->space",SYNC=True)  
        n_objs = reg_boxes.shape[0]
        n_keep = reg_boxes.shape[1]
        cam_names_repeated = [cam for cam in cam_names for i in range(reg_boxes.shape[1])]
        reg_boxes_state = self.hg.im_to_state(reg_boxes.view(-1,8,2),name = cam_names_repeated,classes = classes.view(-1)).view(n_objs,n_keep,-1)
        
 
        # 4. Select best box
        self.tm.split("select_best",SYNC=True)  
        detections, classes, confs = self._select_best_box(objs,reg_boxes_state,confs,classes,n_objs)
        
        
        # note that cam_names and ids directly become detection cameras and detection ids - implicit matching        
        return detections,classes,confs,cam_names,ids
        
        
    def _get_crop_boxes(self,objects):
            """
            Given a set of objects, returns boxes to crop them from the frame
            objects - [n,8,2] array of x,y, corner coordinates for 3D bounding boxes
            
            returns [n,4] array of xmin,xmax,ymin,ymax for cropping each object
            """
            
            # find xmin,xmax,ymin, and ymax for 3D box points
            self.tm.split("Crop:minmax",SYNC=True)
            minx = torch.min(objects[:,:,0],dim = 1)[0]
            miny = torch.min(objects[:,:,1],dim = 1)[0]
            maxx = torch.max(objects[:,:,0],dim = 1)[0]
            maxy = torch.max(objects[:,:,1],dim = 1)[0]
            
            self.tm.split("Crop:scale",SYNC=True)
            w = maxx - minx
            h = maxy - miny
            scale = torch.max(torch.stack([w,h]),dim = 0)[0] * self.box_expansion_ratio
            
            self.tm.split("Crop:xysr",SYNC=True)
            # find a tight box around each object in xysr formulation
            minx2 = (minx+maxx)/2.0 - scale/2.0
            maxx2 = (minx+maxx)/2.0 + scale/2.0
            miny2 = (miny+maxy)/2.0 - scale/2.0
            maxy2 = (miny+maxy)/2.0 + scale/2.0
            
            self.tm.split("Crop:stack",SYNC=True)
            crop_boxes = torch.stack([minx2,miny2,maxx2,maxy2]).transpose(0,1).to(self.device)
            
            # clamp erroneous boxes
            pad = -20
            crop_boxes[:,0] = crop_boxes[:,0].clamp(-pad,1920+pad)
            crop_boxes[:,2] = crop_boxes[:,2].clamp(-pad,1920+pad)
            crop_boxes[:,1] = crop_boxes[:,1].clamp(-pad,1080+pad)
            crop_boxes[:,3] = crop_boxes[:,3].clamp(-pad,1080+pad)

            
            return crop_boxes
        
    def _local_to_global(self,preds,crop_boxes):
        """
        Convert from crop coordinates to frame coordinates
        preds - [n,d,20] array where n indexes object and d indexes detections for that object
        crops_boxes - [n,4] array
        """
        n = preds.shape[0]
        d = preds.shape[1]
        preds = preds.reshape(n,d,10,2)
        preds = preds[:,:,:8,:] # drop 2D boxes
        
        scales = torch.max(torch.stack([crop_boxes[:,2] - crop_boxes[:,0],crop_boxes[:,3] - crop_boxes[:,1]]),dim = 0)[0]
        
        # preds is [n,d,8,2] - expand scale, currently [n], to match
        scales = scales.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,d,8,2)
        
        # scale each box by the box scale / crop size self.cs
        preds = preds * scales / self.crop_size
    
        # shift based on crop box corner
        preds[:,:,:,0] += crop_boxes[:,0].unsqueeze(1).unsqueeze(1).repeat(1,d,8)
        preds[:,:,:,1] += crop_boxes[:,1].unsqueeze(1).unsqueeze(1).repeat(1,d,8)
        
        return preds
    
    def _select_best_box(self,a_priori,preds,confs,classes,n_objs):
        """
        a_priori - [n,6] array of state formulation object priors
        preds    - [n,d,6] array where n indexes object and d indexes detections for that object, in state formulation
        confs   - [n,d] array of confidence for each pred
        classes   - [n,d] array of class prediction for each pred
        returns  - [n,6] array of best matched objects
        """

        # convert preds to roadway coordinate space
        #print(preds.shape,a_priori.shape)
        n,d,_ = preds.shape 
        
        boxes = preds
        #d = boxes.shape[0]
        intermediate_boxes = torch.zeros([n,d,4,2], device = boxes.device)
        intermediate_boxes[:,:,0,0] = boxes[:,:,0] 
        intermediate_boxes[:,:,0,1] = boxes[:,:,1] - boxes[:,:,3]/2.0
        intermediate_boxes[:,:,1,0] = boxes[:,:,0] 
        intermediate_boxes[:,:,1,1] = boxes[:,:,1] + boxes[:,:,3]/2.0
        
        intermediate_boxes[:,:,2,0] = boxes[:,:,0] + boxes[:,:,2]*boxes[:,:,5]
        intermediate_boxes[:,:,2,1] = boxes[:,:,1] + boxes[:,:,2]*boxes[:,:,5] - boxes[:,:,3]/2.0
        intermediate_boxes[:,:,3,0] = boxes[:,:,0] + boxes[:,:,2]*boxes[:,:,5]
        intermediate_boxes[:,:,3,1] = boxes[:,:,1] + boxes[:,:,2]*boxes[:,:,5] + boxes[:,:,3]/2.0

        boxes_new = torch.zeros([n,d,4],device = boxes.device)
        boxes_new[:,:,0] = torch.min(intermediate_boxes[:,:,0:4,0],dim = 2)[0]
        boxes_new[:,:,2] = torch.max(intermediate_boxes[:,:,0:4,0],dim = 2)[0]
        boxes_new[:,:,1] = torch.min(intermediate_boxes[:,:,0:4,1],dim = 2)[0]
        boxes_new[:,:,3] = torch.max(intermediate_boxes[:,:,0:4,1],dim = 2)[0]
        preds_space = boxes_new
        
        
        # convert a priori to roadway coordinate space
        boxes = a_priori
        #n = boxes.shape[0]
        intermediate_boxes = torch.zeros([n,4,2], device = boxes.device)
        intermediate_boxes[:,0,0] = boxes[:,0] 
        intermediate_boxes[:,0,1] = boxes[:,1] - boxes[:,3]/2.0
        intermediate_boxes[:,1,0] = boxes[:,0] 
        intermediate_boxes[:,1,1] = boxes[:,1] + boxes[:,3]/2.0
        
        intermediate_boxes[:,2,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
        intermediate_boxes[:,2,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] - boxes[:,3]/2.0
        intermediate_boxes[:,3,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
        intermediate_boxes[:,3,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] + boxes[:,3]/2.0

        boxes_new = torch.zeros([boxes.shape[0],4],device = boxes.device)
        boxes_new[:,0] = torch.min(intermediate_boxes[:,0:4,0],dim = 1)[0]
        boxes_new[:,2] = torch.max(intermediate_boxes[:,0:4,0],dim = 1)[0]
        boxes_new[:,1] = torch.min(intermediate_boxes[:,0:4,1],dim = 1)[0]
        boxes_new[:,3] = torch.max(intermediate_boxes[:,0:4,1],dim = 1)[0]
        a_priori = boxes_new
        
        # # convert  preds into space 
        # preds_space = self.hg.state_to_space(preds.clone())
        # preds_space = preds_space.reshape(n_objs,-1,8,3)
        # preds = preds.reshape(n_objs,-1,6)
        
        # n = preds_space.shape[0] 
        # d = preds_space.shape[1]
        
        # # convert into xmin ymin xmax ymax form        
        # boxes_new = torch.zeros([n,d,4],device = preds.device)
        # boxes_new[:,:,0] = torch.min(preds_space[:,:,0:4,0],dim = 2)[0]
        # boxes_new[:,:,2] = torch.max(preds_space[:,:,0:4,0],dim = 2)[0]
        # boxes_new[:,:,1] = torch.min(preds_space[:,:,0:4,1],dim = 2)[0]
        # boxes_new[:,:,3] = torch.max(preds_space[:,:,0:4,1],dim = 2)[0]
        # preds_space = boxes_new
        
        # # convert a_priori into space
        # a_priori = self.hg.state_to_space(a_priori.clone())
        # boxes_new = torch.zeros([n,4],device =  a_priori.device)
        # boxes_new[:,0] = torch.min(a_priori[:,0:4,0],dim = 1)[0]
        # boxes_new[:,2] = torch.max(a_priori[:,0:4,0],dim = 1)[0]
        # boxes_new[:,1] = torch.min(a_priori[:,0:4,1],dim = 1)[0]
        # boxes_new[:,3] = torch.max(a_priori[:,0:4,1],dim = 1)[0]
        # a_priori = boxes_new
        
        # a_priori is now [n,4] need to repeat by [d]
        a_priori = a_priori.unsqueeze(1).repeat(1,d,1)
        
        # calculate iou for each
        ious = self._md_iou(preds_space.double(),a_priori.double())
        
        # compute score for each box [n,d]
        scores = (1-self.w) * ious + self.w*confs
        
        keep = torch.argmax(scores,dim = 1)
        
        idx = torch.arange(n)
        best_boxes = preds[idx,keep,:]
        cls_preds = classes[idx,keep]
        confs = confs[idx,keep]
        
        
        # gather max score boxes
        
        return best_boxes, cls_preds, confs
    
    def _md_iou(self,a,b):
        """
        a,b - [batch_size ,num_anchors, 4]
        """
        
        area_a = (a[:,:,2]-a[:,:,0]) * (a[:,:,3]-a[:,:,1])
        area_b = (b[:,:,2]-b[:,:,0]) * (b[:,:,3]-b[:,:,1])
        
        minx = torch.max(a[:,:,0], b[:,:,0])
        maxx = torch.min(a[:,:,2], b[:,:,2])
        miny = torch.max(a[:,:,1], b[:,:,1])
        maxy = torch.min(a[:,:,3], b[:,:,3])
        zeros = torch.zeros(minx.shape,dtype=float,device = a.device)
        
        intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
        union = area_a + area_b - intersection
        iou = torch.div(intersection,union)
        
        #print("MD iou: {}".format(iou.max(dim = 1)[0].mean()))
        return iou