# re-implementation of ngransac as in https://github.com/vislearn/ngransac

import numpy as np
import opensfm
import cv2
from opensfm import dataset, features, io
import json

import torch
import torch.optim as optim
import ngransac

from network import CNNet
from dataset import SparseDataset
import util
import pdb


class NGRansac:
    
    # init model with default params. Calculate only fundamental matrix
    def __init__(self,
                 frame_size,           
                 K = None,
                 batchsize=32,
                 fmat=False,
                 hyps=4000,
                 model='',
                 nfeatures=2000,
                 nosideinfo=False,                 
                 orb=False,
                 ratio=1.0,
                 refine=True,
                 resblocks=12,
                 rootsift=False,
                 session='',
                 threshold=0.001):
        
        print(K)
        if fmat:
            print("\nFitting Fundamental Matrix...\n")
        else:
            print("\nFitting Essential Matrix...\n")
        
        # load network
        model_file = model
        if len(model_file) == 0:
            model_file = util.create_session_string('e2e', fmat, orb, rootsift, ratio, session)
            model_file = '../../ngransac/models/weights_' + model_file + '.net'            
            print('Loading pre-trained model:', model_file)
            
        model = CNNet(resblocks)
        model.load_state_dict(torch.load(model_file))
        model = model.cuda()
        model.eval()        
        print('model ', model_file, ' successfully loaded.')                                    
        
        self.model      = model
        self.frame_size = frame_size
        self.nosideinfo = nosideinfo
        self.hyps       = hyps
        self.threshold  = threshold
        self.refine     = refine
        self.K = K
        
    
    # pass to this function points with their real coordinates
    def findEssentialMat(self, pts1, pts2, ratios):
        
        assert self.K is not None, 'To calculate E, K must be NOT NONE'
        
        pts1 = cv2.undistortPoints(pts1, self.K, None)
        pts2 = cv2.undistortPoints(pts2, self.K, None)
        
        
        if self.nosideinfo:
            # remove side information before passing it to the network
            ratios = np.zeros(ratios.shape)
        
        # create data tensor of feature coordinates and matching ratios
        correspondences = np.concatenate((pts1, pts2, ratios), axis=2)
        correspondences = np.transpose(correspondences)
        correspondences = torch.from_numpy(correspondences).float()

        # predict neural guidance, i.e. RANSAC sampling probabilities
        log_probs = self.model(correspondences.unsqueeze(0).cuda())[0] #zero-indexing creates and removes a dummy batch dimension
        probs = torch.exp(log_probs).cpu()

        out_model     = torch.zeros((3, 3)).float() # estimated model
        out_inliers   = torch.zeros(log_probs.size()) # inlier mask of estimated model
        out_gradients = torch.zeros(log_probs.size()) # gradient tensor (only used during training)
        rand_seed     = 0 # random seed to by used in C++
        
        
        incount = ngransac.find_essential_mat(correspondences, 
                                              probs, 
                                              rand_seed, 
                                              self.hyps, 
                                              self.threshold, 
                                              out_model, 
                                              out_inliers, 
                                              out_gradients)
        
        print("\n=== Model found by NG-RANSAC: =======\n")
        print("\nNG-RANSAC Inliers: ", int(incount))
        
        out_inliers = out_inliers.byte().numpy().ravel().tolist()
        
        # Fundamental matrix
        return out_model.numpy(), out_inliers
    
    # pts1 and pts2 must be normalized to the frame size before running the procedure below
    # as well the ratios must be computed
    def findFundamentalMat(self, pts1, pts2, ratios):
        
        # normalize x and y coordinates before passing them to the network
        # normalized by the image size
        util.normalize_pts(pts1, self.frame_size)
        util.normalize_pts(pts2, self.frame_size)
        
        if self.nosideinfo:
            # remove side information before passing it to the network
            ratios = np.zeros(ratios.shape)
        
        # create data tensor of feature coordinates and matching ratios
        correspondences = np.concatenate((pts1, pts2, ratios), axis=2)
        correspondences = np.transpose(correspondences)
        correspondences = torch.from_numpy(correspondences).float()

        # predict neural guidance, i.e. RANSAC sampling probabilities
        log_probs = self.model(correspondences.unsqueeze(0).cuda())[0] #zero-indexing creates and removes a dummy batch dimension
        probs = torch.exp(log_probs).cpu()

        out_model     = torch.zeros((3, 3)).float() # estimated model
        out_inliers   = torch.zeros(log_probs.size()) # inlier mask of estimated model
        out_gradients = torch.zeros(log_probs.size()) # gradient tensor (only used during training)
        rand_seed     = 0 # random seed to by used in C++
        
        # run NG-RANSAC
        # === CASE FUNDAMENTAL MATRIX =========================================

        # undo normalization of x and y image coordinates
        util.denormalize_pts(correspondences[0:2], self.frame_size)
        util.denormalize_pts(correspondences[2:4], self.frame_size)
        
        incount = ngransac.find_fundamental_mat(correspondences, 
                                                probs, 
                                                rand_seed, 
                                                self.hyps, 
                                                self.threshold, 
                                                self.refine, 
                                                out_model, 
                                                out_inliers, 
                                                out_gradients)
        
        print("\n=== Model found by NG-RANSAC: =======\n")
        print("\nNG-RANSAC Inliers: ", int(incount))
        
        out_inliers = out_inliers.byte().numpy().ravel().tolist()
        
        # Fundamental matrix
        return out_model.numpy(), out_inliers
    
   
