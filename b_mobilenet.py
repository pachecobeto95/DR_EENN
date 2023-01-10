import torch
import torch.nn as nn
#from utils import ExitBlock
from pthflops import count_ops
import torchvision.models as models
import numpy as np
import config
from statistics import mode


class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1,
                 padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class ExitBlock(nn.Module):
  """
  This class defines the Early Exit, which allows to finish the inference at the middle layers when
  the classification confidence achieves a predefined threshold
  """
  def __init__(self, n_classes, input_shape, exit_type, device):
    super(ExitBlock, self).__init__()
    _, channel, width, height = input_shape
    """
    This creates a random input sample whose goal is to find out the input shape after each layer.
    In fact, this finds out the input shape that arrives in the early exits, to build a suitable branch.

    Arguments are

    nIn:          (int)    input channel of the data that arrives into the given branch.
    n_classes:    (int)    number of the classes
    input_shape:  (tuple)  input shape that arrives into the given branch
    exit_type:    (str)    this argument define the exit type: exit with conv layer or not, just fc layer
    dataset_name: (str)   defines tha dataset used to train and evaluate the branchyNet
     """

    self.expansion = 1
    self.device = device
    self.layers = nn.ModuleList()

    # creates a random input sample to find out input shape in order to define the model architecture.
    x = torch.rand(1, channel, width, height).to(device)
    
    self.conv = nn.Sequential(
        ConvBasic(channel, channel, kernel=3, stride=2, padding=1),
        nn.AvgPool2d(2),)
    
    #gives the opportunity to add conv layers in the branch, or only fully-connected layers
    if (exit_type == "conv"):
      self.layers.append(self.conv)
    else:
      self.layers.append(nn.AdaptiveAvgPool2d(2))
      
    feature_shape = nn.Sequential(*self.layers).to(device)(x).shape
    
    total_neurons = feature_shape[1]*feature_shape[2]*feature_shape[3] # computes the input neurons of the fc layer 
    self.layers = self.layers.to(device)
    self.classifier = nn.Linear(total_neurons , n_classes).to(device) # finally creates 
    
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    x = x.view(x.size(0), -1)

    return self.classifier(x)

class B_MobileNet(nn.Module):
  def __init__(self, n_classes: int, 
               pretrained: bool, n_branches: int, img_dim:int, 
               exit_type: str, device, branches_positions=None, distribution="linear"):
    super(B_MobileNet, self).__init__()

    self.n_classes = n_classes
    self.pretrained = pretrained
    self.n_branches = n_branches
    self.img_dim = img_dim
    self.exit_type = exit_type
    self.branches_positions = branches_positions
    self.distribution = distribution
    self.softmax = nn.Softmax(dim=1)
    self.device = device

    self.model = self.initialize_model()
    self.n_blocks = len(list(self.model.features))
    self.insertBranches()
  
  def initialize_model(self):
    model = models.mobilenet_v2(pretrained=self.pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.n_classes)
    return model.to(self.device)
  
  def countFlops(self):
    x = torch.rand(1, 3, self.img_dim, self.img_dim).to(self.device)
    flops_count_dict = {}
    flops_acc_dict = {}
    flops_list = []
    total_flops = 0
    for i, layer in enumerate(self.model.features, 1):
      ops, all_data = count_ops(layer, x, print_readable=False, verbose=False)
      x = layer(x)
      flops_count_dict[i] = ops
      total_flops += ops
      flops_acc_dict[i] = total_flops
    
    #for key, value in flops_acc_dict.items():
    #  flops_acc_dict[key] = value/total_flops

    return flops_count_dict, flops_acc_dict, total_flops

  def set_thresholds(self, total_flops):
    """
    """
    gold_rate = 1.61803398875
    flop_margin = 1.0 / (self.n_branches+1)
    self.threshold = []
    self.percentage_threshold = []
        
    for i in range(self.n_branches):
      if (self.distribution == 'pareto'):
        self.threshold.append(total_flops * (1 - (0.8**(i+1))))
        self.percentage_threshold.append(1 - (0.8**(i+1)))
      elif (self.distribution == 'fine'):
        self.threshold.append(total_flops * (1 - (0.95**(i+1))))
        self.percentage_threshold.append(1 - (0.95**(i+1)))
      elif (self.distribution == 'linear'):
        self.threshold.append(total_flops * flop_margin * (i+1))
        self.percentage_threshold.append(flop_margin * (i+1))

      else:
        self.threshold.append(total_flops * (gold_rate**(i - self.num_ee)))
        self.percentage_threshold.append(gold_rate**(i - self.n_branches))
  
  
  def is_suitable_for_exit(self, i, flop_count):
    if (self.branches_positions is None):
      return self.stage_id < self.n_branches and flop_count >= self.threshold[self.stage_id]
    
    else:
      return i in self.branches_positions
  
  def add_early_exit(self, layer):
    #print("Adding")
    self.stages.append(nn.Sequential(*self.layers))
    x = torch.rand(1, 3, self.img_dim, self.img_dim).to(self.device)
    feature_shape = nn.Sequential(*self.stages)(x).shape
    self.exits.append(ExitBlock(self.n_classes, feature_shape, self.exit_type, self.device))
    self.stage_id += 1
    self.layers = nn.ModuleList()

  def insertBranches(self):
    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.stage_id = 0

    flops_count_dict, flops_acc_dict, total_flops = self.countFlops()
    self.set_thresholds(total_flops)

    for i, layer in enumerate(self.model.features, 1):
      if (self.is_suitable_for_exit(i, flops_acc_dict[i])):
        self.add_early_exit(layer)
      else:
        self.layers.append(layer)

    self.stages.append(nn.Sequential(*self.layers))
    self.fully_connected = self.model.classifier


  def forwardTrain(self, x):

    conf_list, class_list  = [], []
    #inference_time_list = []

    #cumulative_inference_time = 0.0

    #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for i, exitBlock in enumerate(self.exits):
      
      #starter.record()

      x = self.stages[i](x)
      output_branch = exitBlock(x)

      prob_vector = self.softmax(output_branch)
      conf, infered_class = torch.max(prob_vector, 1)
            
      #ender.record()
      #torch.cuda.synchronize()
      #curr_time = starter.elapsed_time(ender)

      #cumulative_inference_time += curr_time

      #inference_time_list.append(cumulative_inference_time)

      conf_list.append(conf), class_list.append(infered_class-1)#, prob_vector_list.append(prob_vector)

    #starter.record()

    x = self.stages[-1](x)
    x = x.mean(3).mean(2)

    output = self.fully_connected(x)
    prob_vector = self.softmax(output)

    #ender.record()
    #torch.cuda.synchronize()
    #curr_time = starter.elapsed_time(ender)

    #cumulative_inference_time += curr_time

    #inference_time_list.append(cumulative_inference_time)

    infered_conf, infered_class = torch.max(prob_vector, 1)
    
    conf_list.append(infered_conf), class_list.append(infered_class-1)#, prob_vector_list.append(prob_vector.cpu().numpy().reshape(self.n_classes))
    #prob_vector_list.append(prob_vector)

    #return prob_vector_list, conf_list, class_list, inference_time_list
    return conf_list, class_list

  def forwardEval(self, x, p_tar):
    output_list, conf_list, class_list  = [], [], []
    for i, exitBlock in enumerate(self.exits): #[:config.N_BRANCHES] it acts to select until branches will be processed. 
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      conf, infered_class = torch.max(self.softmax(output_branch), 1)
      if (conf.item() > p_tar):
        return output_branch, conf_list, infered_class

      else:
        output_list.append(output_branch)
        conf_list.append(conf.item())
        class_list.append(infered_class)

    return x, conf_list, None

  def forwardEmulation(self, x, p_tar_list):
    output_list, conf_list, class_list  = [], [], []
    for i, exitBlock in enumerate(self.exits): #[:config.N_BRANCHES] it acts to select until branches will be processed. 
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      conf, infered_class = torch.max(self.softmax(output_branch), 1)
      if (conf.item() > p_tar_list[i]):
        return output_branch, conf_list, infered_class

      else:
        output_list.append(output_branch)
        conf_list.append(conf.item())
        class_list.append(infered_class)



    return x, conf_list, None

  def forwardExp(self, x, p_tar):
    output_list, conf_list, class_list  = [], [], []
    for i, exitBlock in enumerate(self.exits): #[:config.N_BRANCHES] it acts to select until branches will be processed. 
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      conf, infered_class = torch.max(self.softmax(output_branch), 1)
      #print(conf)
      if (conf.item() > p_tar):
      #  print("early")
        return output_branch, conf_list, infered_class, True

      else:
        output_list.append(output_branch)
        conf_list.append(conf.item())
        class_list.append(infered_class)

    return x, conf_list, class_list, False


  def forwardEeInference(self, x, nr_branch_edge, p_tar):

    conf_list, class_list = [], []
    n_exits = self.n_branches + 1

    #print(x.shape)

    #for i, exitBlock in enumerate(self.exits):
    for i, exitBlock in enumerate(self.exits[:nr_branch_edge]):
      #print(i)
      x = self.stages[i](x)

      output_branch = exitBlock(x)
      conf_branch, infered_class_branch = torch.max(self.softmax(output_branch), 1)

      if (conf_branch.item() >= p_tar):
        return output_branch, conf_branch.item(), infered_class_branch.item(), True

      else:
        conf_list.append(conf_branch.item()), class_list.append(infered_class_branch.item())
      
    return x, conf_list, class_list, False

  def forwardEnsembleInference(self, x, acc_branches, nr_branch_edge, p_tar, device):

    nr_classes = config.nr_class_dict["caltech256"]

    ensemble_prob_vector = torch.zeros((1, nr_classes), device=device)

    conf_list, class_list = [], []
    n_exits = self.n_branches + 1

    for i, exitBlock in enumerate(self.exits[:nr_branch_edge]):

      x = self.stages[i](x)

      output_branch = exitBlock(x)

      prob_vector = self.softmax(output_branch)

      ensemble_prob_vector += acc_branches[i]*prob_vector

    ensemble_conf, ensemble_infered_class = torch.max(ensemble_prob_vector, 1)

    ensemble_infered_class -= 1 
  
    #correct = ensemble_infered_class.eq(target.view_as(ensemble_infered_class)).sum().item()

    wasClassified = True if(ensemble_conf.item() >= p_tar) else False

    return x, [ensemble_conf.item()], [ensemble_infered_class.item()], wasClassified


  def forwardNaiveEnsembleInference(self, x, nr_branch_edge, p_tar):

    conf_list, class_list = [], []

    for i, exitBlock in enumerate(self.exits[:nr_branch_edge]):

      x = self.stages[i](x)

      output_branch = exitBlock(x)
      conf_branch, infered_class_branch = torch.max(self.softmax(output_branch), 1)

      conf_list.append(conf_branch.item()), class_list.append(infered_class_branch.item())


    max_conf_idx = np.argmax(conf_list)
    #mode_list = multimode(class_list)
    mode_list = max(set(class_list), key = class_list.count)
    
    ensemble_conf = conf_list[max_conf_idx]

    #print(ensemble_conf)

    wasClassified = True if(ensemble_conf >= p_tar) else False
      
    return x, [ensemble_conf], [mode_list], wasClassified


  def forwardCoEeInferenceCloud(self, x, conf_list, class_list, p_tar, n_branch_edge):


    for i, exitBlock in enumerate(self.exits[n_branch_edge:], n_branch_edge): #[config.n_branch_edge:] it acts to select until branches will be processed. 
      
      x = self.stages[i](x)

    x = self.stages[-1](x)
    x = x.mean(3).mean(2)

    output = self.fully_connected(x)
    conf, infered_class = torch.max(self.softmax(output), 1)
    
    conf_list.append(conf.item())
    class_list.append(infered_class.item())
    
    if (conf.item() >= p_tar):
      return conf, infered_class
    
    else:
      max_conf = np.argmax(conf_list)
      
      return conf_list[max_conf], class_list[max_conf]


  def forwardEnsembleInferenceCloud(self, x, conf_list, class_list, p_tar, n_branch_edge):


    output_list = []

    for i, exitBlock in enumerate(self.exits[n_branch_edge:], n_branch_edge): #[config.n_branch_edge:] it acts to select until branches will be processed. 
      
      x = self.stages[i](x)


    x = self.stages[-1](x)
    x = x.mean(3).mean(2)

    output = self.fully_connected(x)
    conf, infered_class = torch.max(self.softmax(output), 1)
    
    conf_list.append(conf.item())
    class_list.append(infered_class.item())
    
    if (conf.item() >= p_tar):
      return conf, infered_class
    
    else:
      max_conf = np.argmax(conf_list)
      return conf_list[max_conf], class_list[max_conf]


  def forwardNaiveEnsembleInferenceCloud(self, x, conf_list, class_list, p_tar, n_branch_edge):


    output_list = []

    for i, exitBlock in enumerate(self.exits[n_branch_edge:], n_branch_edge): #[config.n_branch_edge:] it acts to select until branches will be processed. 
      
      x = self.stages[i](x)


    x = self.stages[-1](x)
    x = x.mean(3).mean(2)

    output = self.fully_connected(x)
    conf, infered_class = torch.max(self.softmax(output), 1)
    
    conf_list.append(conf.item())
    class_list.append(infered_class)
    
    if (conf.item() >= p_tar):
      return conf, infered_class
    
    else:
      #print(conf_list)
      max_conf = np.argmax(conf_list)
      return conf_list[max_conf], class_list[max_conf]



  def forward(self, x):
    return self.forwardTrain(x)
