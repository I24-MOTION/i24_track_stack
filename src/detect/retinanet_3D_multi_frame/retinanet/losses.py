import numpy as np
import torch
import torch.nn as nn


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(
        a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(
        a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) *
                         (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations, embeddings):
        top_weighting = 0.5
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        vp_losses = []
        embedding_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        flat_embeddings = []
        flat_embedding_targets = []
        # separate vp terms
        #annotations = annotations[:,:,:21]

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            embedding = embeddings[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(
                        classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * \
                        torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * \
                        torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

            # here, we need a custom bit to calculate 2D bbox based on 3D bbox
            # actually, instead we now expect a 2D box in  last 4 positions, and 3D corners in last 16
            #bbox_annotation_2D = bbox_annotation[:,-4:]

            xmin, _ = torch.min(bbox_annotation[:, [0, 2, 4, 6]], dim=1)
            xmax, _ = torch.max(bbox_annotation[:, [0, 2, 4, 6]], dim=1)
            ymin, _ = torch.min(bbox_annotation[:, [1, 3, 5, 7]], dim=1)
            ymax, _ = torch.max(bbox_annotation[:, [1, 3, 5, 7]], dim=1)

            xmin2, _ = torch.min(bbox_annotation[:, [8, 10, 12, 14]], dim=1)
            xmax2, _ = torch.max(bbox_annotation[:, [8, 10, 12, 14]], dim=1)
            ymin2, _ = torch.min(bbox_annotation[:, [9, 11, 13, 15]], dim=1)
            ymax2, _ = torch.max(bbox_annotation[:, [9, 11, 13, 15]], dim=1)

            xmin = torch.min(xmin, xmin2).unsqueeze(1)
            xmax = torch.max(xmax, xmax2).unsqueeze(1)
            ymin = torch.min(ymin, ymin2).unsqueeze(1)
            ymax = torch.max(ymax, ymax2).unsqueeze(1)
            bbox_annotation_2D = torch.cat((xmin, ymin, xmax, ymax), dim=1)

            # num_anchors x num_annotations
            IoU = calc_iou(anchors[0, :, :], bbox_annotation_2D)
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            #import pdb
            # pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            # here we select the last element from the assigned annotation and set that class as positive
            targets[positive_indices,
                    assigned_annotations[positive_indices, -1].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(
                torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(
                torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) +
                    (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(
                    torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(
                    torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(
                cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                # HERE, we'll need to redefine this logic for 3D bbox formulation
                # normalize coordinates by width and height
                # normalize tail length by dividing then taking log

                # gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                # gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                # gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                # gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # # clip widths to 1
                # gt_widths  = torch.clamp(gt_widths, min=1)
                # gt_heights = torch.clamp(gt_heights, min=1)

                # targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                # targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                # targets_dw = torch.log(gt_widths / anchor_widths_pi)
                # targets_dh = torch.log(gt_heights / anchor_heights_pi)

                # targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                # targets = targets.t()
                # remove ids and classifications from targets
                targets = assigned_annotations[:, :-2]

                # regression is x,y,lx,ly,wx,wy,hx,hy - need to convert to corner coordinates
                # fbl fbr bbl bbr ftl ftr btl btr - alternate w first, then l, then h

                # [n_positive_indices,12]
                regression = regression[positive_indices, :]

                # REID Loss

                # target ids = [n_annotations,1]
                target_ids = assigned_annotations[:, -2]
                
                # first, get the associated predicted vectors which should be [n_annotations, embedding_size]
                embedding = embedding[positive_indices, :]

                flat_embeddings.append(embedding)
                flat_embedding_targets.append(target_ids)
                
                
                
                
                # print(regression.shape)
                # print(embedding.shape)
                # print(target_ids.shape)
                
                
                

                # VP vectors are computed from object center to vanishing point
                # Object vectors are computed towards the back, towards the right, and towards the top
                # Thus, if the vanishing point is closer to the back/right/top, we want the angle to be 0 (cos term = 1)
                # Otherwise, we want the angle to be 180 (cos term = 1)

                # we can look at the assigned annotation for each prediction, and compute a scale factor (1 or -1) based on the box's orientation
                # then multiply the cos terms by this factor at the end

                # VP 1
                # we compute the line from each box towards each vp direction
                # vector components
                reg_vec_x = regression[:, 2]
                reg_vec_y = regression[:, 3]

                # vector is in direction front -> back
                targ_vec_x = ((targets[:, 4] + targets[:, 6] + targets[:, 12] + targets[:, 14]) - (
                    targets[:, 0] + targets[:, 2] + targets[:, 8] + targets[:, 10]))/4.0
                targ_vec_y = ((targets[:, 5] + targets[:, 7] + targets[:, 13] + targets[:, 15]) - (
                    targets[:, 1] + targets[:, 3] + targets[:, 9] + targets[:, 11]))/4.0

                # dot product
                reg_norm = torch.sqrt(
                    torch.pow(reg_vec_x, 2) + torch.pow(reg_vec_y, 2))
                targ_norm = torch.sqrt(
                    torch.pow(targ_vec_x, 2) + torch.pow(targ_vec_y, 2))
                cos_angle = (reg_vec_x * targ_vec_x + reg_vec_y *
                             targ_vec_y)/(reg_norm * targ_norm)

                # based on new definition, don't need the sign term
                # # sign term
                # # distance of front bottom left from vp
                # d1 = torch.sqrt((vp[:,0] - targets[:,0])**2 + (vp[:,1] - targets[:,1])**2)
                # # distance of back bottom left from vp
                # d2 = torch.sqrt((vp[:,0] - targets[:,4])**2 + (vp[:,1] - targets[:,5])**2)
                # # if back is closer than front (d1-d2) + , reg_vec points towards vp (sign +)
                # sign_vec = torch.sign(d1-d2)
                # multiply loss term by sign

                vp1_loss = 1 - cos_angle

                # for our loss term we'll use 1-cos(angle) = 1- vec1 . vec2 / (||vec1||*||vec2||)
                # we have to consider both reg vector orientations and take best
                #vp1_loss = 1-torch.pow(cos_angle,2)

                # VP 2
                # we compute the line from each box towards each vp direction
                # vector components
                reg_vec_x = regression[:, 4]
                reg_vec_y = regression[:, 5]

                # vector is in direction left -> right (so add right side, then subtract left side)
                targ_vec_x = ((targets[:, 2] + targets[:, 6] + targets[:, 10] + targets[:, 14]) - (
                    targets[:, 0] + targets[:, 4] + targets[:, 8] + targets[:, 12]))/4.0
                targ_vec_y = ((targets[:, 3] + targets[:, 7] + targets[:, 11] + targets[:, 15]) - (
                    targets[:, 1] + targets[:, 5] + targets[:, 9] + targets[:, 13]))/4.0

                # dot product
                reg_norm = torch.sqrt(
                    torch.pow(reg_vec_x, 2) + torch.pow(reg_vec_y, 2))
                targ_norm = torch.sqrt(
                    torch.pow(targ_vec_x, 2) + torch.pow(targ_vec_y, 2))
                cos_angle = (reg_vec_x * targ_vec_x + reg_vec_y *
                             targ_vec_y)/(reg_norm * targ_norm)

                # # sign term
                # # distance of front bottom left from vp
                # d1 = torch.sqrt((vp[:,2] - targets[:,0])**2 + (vp[:,3] - targets[:,1])**2)
                # # distance of front bottom right from vp
                # d2 = torch.sqrt((vp[:,2] - targets[:,2])**2 + (vp[:,3] - targets[:,3])**2)
                # # sign should be positive if right is closer
                # sign_vec = torch.sign(d1-d2)
                # # multiply loss term by sign
                vp2_loss = 1 - cos_angle

                # for our loss term we'll use 1-cos(angle) = 1- vec1 . vec2 / (||vec1||*||vec2||)
                # we have to consider both reg vector orientations and take best
                #vp2_loss = 1-torch.pow(cos_angle,2)

                # VP 3
                # we compute the line from each box towards each vp direction
                # vector_components
                reg_vec_x = regression[:, 6]
                reg_vec_y = regression[:, 7]

                # vector is in direction top -> bottom (so add bottom, then subtract top)
                targ_vec_x = ((targets[:, 0] + targets[:, 2] + targets[:, 4] + targets[:, 6]) - (
                    targets[:, 8] + targets[:, 10] + targets[:, 12] + targets[:, 14]))/4.0
                targ_vec_y = ((targets[:, 1] + targets[:, 3] + targets[:, 5] + targets[:, 7]) - (
                    targets[:, 9] + targets[:, 11] + targets[:, 13] + targets[:, 15]))/4.0

                # dot product
                reg_norm = torch.sqrt(
                    torch.pow(reg_vec_x, 2) + torch.pow(reg_vec_y, 2))
                targ_norm = torch.sqrt(
                    torch.pow(targ_vec_x, 2) + torch.pow(targ_vec_y, 2))
                cos_angle = (reg_vec_x * targ_vec_x + reg_vec_y *
                             targ_vec_y)/(reg_norm * targ_norm)

                # # sign term
                # # distance of front bottom left from vp
                # d1 = torch.sqrt((vp[:,4] - targets[:,0])**2 + (vp[:,5] - targets[:,1])**2)
                # # distance of front top left from vp
                # d2 = torch.sqrt((vp[:,4] - targets[:,8])**2 + (vp[:,5] - targets[:,9])**2)
                # # if bottom is closer (d2-d1) +, sign should be +
                # sign_vec = torch.sign(d2-d1)
                # # multiply loss term by sign
                vp3_loss = 1 - cos_angle

                # for our loss term we'll use 1-cos(angle) = 1- vec1 . vec2 / (||vec1||*||vec2||)
                # we have to consider both reg vector orientations and take best
                #vp3_loss = 1-torch.pow(cos_angle,2)

                vp_loss = (vp1_loss + vp2_loss + vp3_loss)/3.0
                vp_losses.append(vp_loss.mean())

                # THE ACTUAL REGRESSION LOSS

                # try to introduce bias so all directions are equally possible anglewise /???
                #regression[:,2:] -= 0.5

                preds = torch.zeros([regression.shape[0], 20],
                                    requires_grad=True).cuda()
                preds[:, 0] = regression[:, 0] - regression[:, 2] - \
                    regression[:, 4] + regression[:, 6]
                preds[:, 1] = regression[:, 1] - regression[:, 3] - \
                    regression[:, 5] + regression[:, 7]
                preds[:, 2] = regression[:, 0] - regression[:, 2] + \
                    regression[:, 4] + regression[:, 6]
                preds[:, 3] = regression[:, 1] - regression[:, 3] + \
                    regression[:, 5] + regression[:, 7]
                preds[:, 4] = regression[:, 0] + regression[:, 2] - \
                    regression[:, 4] + regression[:, 6]
                preds[:, 5] = regression[:, 1] + regression[:, 3] - \
                    regression[:, 5] + regression[:, 7]
                preds[:, 6] = regression[:, 0] + regression[:, 2] + \
                    regression[:, 4] + regression[:, 6]
                preds[:, 7] = regression[:, 1] + regression[:, 3] + \
                    regression[:, 5] + regression[:, 7]

                preds[:, 8] = regression[:, 0] - regression[:, 2] - \
                    regression[:, 4] - regression[:, 6]
                preds[:, 9] = regression[:, 1] - regression[:, 3] - \
                    regression[:, 5] - regression[:, 7]
                preds[:, 10] = regression[:, 0] - regression[:, 2] + \
                    regression[:, 4] - regression[:, 6]
                preds[:, 11] = regression[:, 1] - regression[:, 3] + \
                    regression[:, 5] - regression[:, 7]
                preds[:, 12] = regression[:, 0] + regression[:, 2] - \
                    regression[:, 4] - regression[:, 6]
                preds[:, 13] = regression[:, 1] + regression[:, 3] - \
                    regression[:, 5] - regression[:, 7]
                preds[:, 14] = regression[:, 0] + regression[:, 2] + \
                    regression[:, 4] - regression[:, 6]
                preds[:, 15] = regression[:, 1] + regression[:, 3] + \
                    regression[:, 5] - regression[:, 7]
                preds[:, 16:20] = regression[:, 8:12]

                targets[:, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]] = (
                    targets[:, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]] - anchor_ctr_x_pi.unsqueeze(1).repeat(1, 10)) / anchor_widths_pi.unsqueeze(1).repeat(1, 10)
                targets[:, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]] = (
                    targets[:, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]] - anchor_ctr_y_pi.unsqueeze(1).repeat(1, 10)) / anchor_heights_pi.unsqueeze(1).repeat(1, 10)

                # std_dev
                #targets = targets/(0.1*torch.ones([10]).cuda())

                negative_indices = 1 + (~positive_indices)

                #regression_diff = torch.abs(targets - regression[positive_indices, :])
                regression_diff = torch.abs(targets - preds)

                # here, we underweight the top corner coords by a factor of self.top_weighting
                regression_diff[:, 8:16] *= top_weighting

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    vp_losses.append(torch.tensor(0).float().cuda())
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    vp_losses.append(torch.tensor(0).float())
                    regression_losses.append(torch.tensor(0).float())


        # outside of the batch_index loop, deal with all embeddings
        if len(flat_embeddings) > 0 and len(flat_embedding_targets) > 0:
            flat_embeddings = torch.cat(flat_embeddings,dim = 0)
            flat_embedding_targets = torch.cat(flat_embedding_targets,dim = 0)
            es = flat_embedding_targets.shape[0]
            emb_size = flat_embeddings.shape[-1]
        
        
            et1 = flat_embedding_targets.unsqueeze(0).expand(es,es)
            et = (et1 == et1.transpose(0,1)).int() 
            
            pos_sum = torch.sum(et) - es # since diagonal is 1 but shouldn't be counted
            neg_sum = torch.sum (1-et) 
            
            pos_ratio = pos_sum / neg_sum
            neg_ratio = 1 - pos_ratio + 1e-04
            
            pos_mask = et/pos_ratio
            neg_mask = (1-et)/neg_ratio
            
            mask = pos_mask + neg_mask
            mask = mask * (1-torch.eye(es).cuda())
            
            et = et*2 -1
            exp_embeddings = flat_embeddings.unsqueeze(0).expand(es,es,emb_size)
            emb_loss = torch.nn.functional.cosine_embedding_loss(exp_embeddings.reshape(-1,emb_size),exp_embeddings.transpose(0,1).reshape(-1,emb_size),et.reshape(-1),margin =-.2,reduction = "none")
            weighted = mask.view(-1) * emb_loss
            emb_loss = weighted.cuda()
        else:
            emb_loss = torch.zeros(1).float().cuda()
        
        
        # summarize embedding score in understandable terms
        if torch.rand(1).item() < 0.01:
            embedding_distance = torch.cosine_similarity(exp_embeddings.reshape(-1,emb_size),exp_embeddings.transpose(0,1).reshape(-1,emb_size), dim = 1)
            hits = embedding_distance[et.reshape(-1) == 1]
            misses = embedding_distance[et.reshape(-1) == -1]
            res = "{:.3f} SAME: {:.3f} ({:.3f})stdv    DIFFERENT: {:.3f} ({:.3f})stdv".format(pos_ratio,hits.mean(),hits.std(),misses.mean(),misses.std())
            print(res)
        
            with open("train_monitor.txt", "a") as text_file:
                text_file.write(res + "\n")
        
        classification_losses = [item.cuda() for item in classification_losses]
        regression_losses = [item.cuda() for item in regression_losses]
        vp_losses = [item.cuda() for item in vp_losses]
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True), torch.stack(vp_losses).mean(dim=0, keepdim=True), emb_loss.mean(dim = 0,keepdim = True)
