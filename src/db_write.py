from i24_database_api.db_writer import DBWriter
from i24_configparse import parse_cfg
from i24_logger.log_writer import catch_critical,logger

import os
import numpy as np


class WriteWrapper():
    
    @catch_critical()
    def __init__(self,collection_overwrite = None, server_id = -1):
    
        
    
        self.SESSION_CONFIG_ID = os.environ["TRACK_CONFIG_SECTION"]
        self.PID = os.getpid()
        self.COMPUTE_NODE_ID = server_id 
        
        self = parse_cfg("TRACK_CONFIG_SECTION",obj=self)    
    
        
        # self.dbw = DBWriter(
        #                host               = self.host, 
        #                port               = self.port, 
        #                username           = self.username, 
        #                password           = self.password,
        #                database_name      = self.db_name, 
        #                schema_file        = self.schema_file,
        #                collection_name    = self.raw_collection,
        #                server_id          = self.COMPUTE_NODE_ID, 
        #                session_config_id  = self.SESSION_CONFIG_ID,
        #                process_id         = self.PID,
        #                process_name       = "track"
        #                )
    
        if collection_overwrite is not None:
            self.raw_collection = collection_overwrite
    
        param = {
          "default_host": self.host,
          "default_port": self.port,
          "default_username": self.username,
          "readonly_user":self.username,
          "default_password": self.password,
          "db_name": self.db_name,
          "collection_name":self.raw_collection,
          "schema_file": self.schema_file,
          "server_id": self.COMPUTE_NODE_ID, 
          "session_config_id": self.SESSION_CONFIG_ID,
          "process_name": "track"
        }
        
        
        
        self.dbw = DBWriter(param,collection_name = self.raw_collection)
        logger.debug("Initialized db writer to Collection {} ({} existing records)".format(self.raw_collection,len(self)))
        
        self.prev_len = len(self) -1
        self.prev_doc = None
        
    
    @catch_critical()
    def __len__(self):
        return self.dbw.collection.count_documents({})
    
    
    @catch_critical()
    def insert(self,trajectories,time_offset = 0):
        """
        Converts trajectories as dequeued from TrackState into document form and inserts with dbw
        trajectories - output from TrackState.remove()
        """
        
        if len(trajectories) == 0:
            return
        
        # cur_len = len(self)
        # if cur_len == self.prev_len:
        #     logger.warning("\n Document {} was not correctly inserted into database collection {}".format(self.prev_doc,self.raw_collection))
        # self.prev_len = cur_len
        
        for id in trajectories.keys():
            trajectory = trajectories[id]
            history = trajectory[0]
            cls_data = trajectory[1]
            
            
            cls = int(np.argmax(cls_data))
            direction = -1 if y[0] > 60 else 1
            timestamps = [item[0] + time_offset for item in history]
            x = [item[1][0].item()  for item in history]
            y = [item[1][1].item()  for item in history]
            l = [item[1][2].item()  for item in history]
            w = [item[1][3].item()  for item in history]
            h = [item[1][4].item()  for item in history]
    
            # convert to document form
            doc = {}
            doc["configuration_id"]        = self.SESSION_CONFIG_ID
            doc["local_fragment_id"]       = id
            doc["compute_node_id"]         = self.COMPUTE_NODE_ID
            doc["coarse_vehicle_class"]    = cls
            doc["fine_vehicle_class"]      = -1
            doc["timestamp"]               = timestamps
            doc["raw timestamp"]           = timestamps
            doc["first_timestamp"]         = timestamps[0]
            doc["last_timestamp"]          = timestamps[-1]
            doc["road_segment_ids"]        = [-1]
            doc["x_position"]              = x
            doc["y_position"]              = y
            doc["starting_x"]              = x[0]
            doc["ending_x"]                = x[-1]
            doc["camera_snapshots"]        = "None"
            doc["flags"]                   = ["test flag 1","test flag 2"]
            doc["length"]                  = l
            doc["width"]                   = w
            doc["height"]                  = h
            doc["direction"]               = direction
            
            
            
            
            # insert
            if len(x) > self.min_document_length:
                self.dbw.write_one_trajectory(**doc) 
                
class WriteWrapperConf():
    
    def __init__(self,collection_overwrite = None, server_id = -1,mm_offset = 0):
    
        
        self.SESSION_CONFIG_ID = -1 #os.environ["TRACK_CONFIG_SECTION"]
        self.PID = os.getpid()
        self.COMPUTE_NODE_ID = server_id 
        self.mm_offset = mm_offset
        
        self = parse_cfg("TRACK_CONFIG_SECTION",obj=self)    
    
        
        # self.dbw = DBWriter(
        #                host               = self.host, 
        #                port               = self.port, 
        #                username           = self.username, 
        #                password           = self.password,
        #                database_name      = self.db_name, 
        #                schema_file        = self.schema_file,
        #                collection_name    = self.raw_collection,
        #                server_id          = self.COMPUTE_NODE_ID, 
        #                session_config_id  = self.SESSION_CONFIG_ID,
        #                process_id         = self.PID,
        #                process_name       = "track"
        #                )
    
        if collection_overwrite is not None:
            self.raw_collection = collection_overwrite
    
        param = {
          "default_host": self.host,
          "default_port": self.port,
          "default_username": self.username,
          "readonly_user":self.username,
          "default_password": self.password,
          "db_name": self.db_name,
          "collection_name":self.raw_collection,
          "schema_file": self.schema_file,
          "server_id": self.COMPUTE_NODE_ID, 
          "session_config_id": self.SESSION_CONFIG_ID,
          "process_name": "track"
        }
        
        
        
        self.dbw = DBWriter(param,collection_name = self.raw_collection,schema_file = self.schema_file)
        #logger.debug("Initialized db writer to Collection {} ({} existing records)".format(self.raw_collection,len(self)))
        
        self.prev_len = len(self) -1
        self.prev_doc = None
        
    
    @catch_critical()
    def __len__(self):
        return self.dbw.collection.count_documents({})
    
    def insert(self,trajectories,cause_of_death = None,time_offset = 0):
        """
        Converts trajectories as dequeued from TrackState into document form and inserts with dbw
        trajectories - output from TrackState.remove()
        cause_of_death - list of str of same length as trajectories
        """
        
        if len(trajectories) == 0:
            return
        
        # cur_len = len(self)
        # if cur_len == self.prev_len:
        #     logger.warning("\n Document {} was not correctly inserted into database collection {}".format(self.prev_doc,self.raw_collection))
        # self.prev_len = cur_len
        
        # optionally de-bias states before writing
        
        flip  = -1
        
        ts_bias = 0
        if self.ts_bias:
            #ts_bias = -0.0275 #-0.0333
            #ts_bias = -0.003
            pass
        bias = np.zeros([6])
        if self.state_bias:
            bias = np.array([0, -0.2, -1.5, -0.4, 0.3])  
        
                
        for id in trajectories.keys():
            trajectory = trajectories[id]
            history = trajectory[0]
            cls_data = trajectory[1]
            conf_data = trajectory[2]
            
            direction = - 1 if history[0][1][1].item() > 0  else 1
            
            cls = int(np.argmax(cls_data))
            timestamps = [item[0] + time_offset + ts_bias for item in history]
            x = [item[1][0].item() + (bias[0]*direction) + self.mm_offset for item in history]
            y = [flip * item[1][1].item() + bias[1] for item in history]
            l = [item[1][2].item() + bias[2] for item in history]
            w = [item[1][3].item() + bias[3] for item in history]
            h = [item[1][4].item() + bias[4] for item in history]
            v = [item[1][5].item()            for item in history]     
            posterior_covariance = (np.stack([item[2].data.numpy() for item in history])[:,:6]).tolist()
            detections = (np.stack([item[3].data.numpy() for item in history])[:,:5]).tolist()
            prior_covariance = (np.stack([item[4].data.numpy() for item in history])[:,:6]).tolist()
            prior = (np.stack([item[5].data.numpy() for item in history])[:,:6]).tolist()
            confs = [conf.item() for conf in conf_data]
            
            # new, more failsafe definition for direction
            direction = 1 if (x[-1] - x[0]) > 0 else -1
            
            # convert to document form
            doc = {}
            doc["configuration_id"]        = self.SESSION_CONFIG_ID
            doc["local_fragment_id"]       = id
            doc["compute_node_id"]         = self.COMPUTE_NODE_ID
            doc["coarse_vehicle_class"]    = cls
            doc["fine_vehicle_class"]      = -1
            doc["timestamp"]               = timestamps
            doc["raw timestamp"]           = timestamps
            doc["first_timestamp"]         = timestamps[0]
            doc["last_timestamp"]          = timestamps[-1]
            doc["road_segment_ids"]        = [-1]
            doc["x_position"]              = x
            doc["y_position"]              = y
            doc["starting_x"]              = x[0]
            doc["ending_x"]                = x[-1]
            doc["camera_snapshots"]        = "None"
            doc["flags"]                   = ["test flag 1","test flag 2"]
            doc["length"]                  = l
            doc["width"]                   = w
            doc["height"]                  = h
            doc["velocity"]                = v
            doc["direction"]               = direction
            doc["detection_confidence"]    = confs
            doc["posterior_covariance"]    = posterior_covariance
            doc["detection"]               = detections
            doc["prior_covariance"]        = prior_covariance
            doc["prior"]                   = prior
            
            if cause_of_death is not None:
                doc["flags"]                   = [cause_of_death[id]]
            
            # insert
            if len(x) > self.min_document_length:
                self.dbw.write_one_trajectory(**doc) 

            
if __name__ == "__main__":
    test = WriteWrapperConf()