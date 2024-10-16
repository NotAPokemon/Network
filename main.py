import random
import json
import numpy as np
from PIL import Image





def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def image_to_normalized_grayscale_list(image_path):
  with Image.open(image_path) as img:
      img = img.convert('L')

      img_array = np.array(img)

      normalized_array = img_array / 255.0
      normalized_array = (normalized_array * 254) + 1

      flattened_array = normalized_array.flatten()

      img_list = flattened_array.tolist()

      return img_list

image_path = 'cobble.png'
pixel_list = image_to_normalized_grayscale_list(image_path)

if True:
  charlist = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","0","1","2","3","4","5","6","7","8","9"]

def Uuid(len):
  res = ""
  for i in range(len):
    res = res + random.choice(charlist)
    res = res + str(i)
  return res

class Node:
  def __init__(self, inputamount):
    self.amount = inputamount
    self.values = [False]
    self.uuid = Uuid(64)
    
  def add(self, value, w):
    if self.values[0]:
      return
    self.values.append(value * w)
    if len(self.values) - 1 == self.amount:
      self.combine()
      
  def combine(self):
    if self.values[0]:
      return
    val = 0
    for i in range(self.amount):
      val = val + self.values[i + 1]
    self.values = [True,max(val*0.001,val)]

  def __str__(self):
    if self.values[0]:
      return str(self.values[1])
    else:
      return "Error: Node has not been calculated yet"
  def __int__(self):
    if self.values[0]:
      return int(self.values[1])
    else:
      return "Error: Node has not been calculated yet"
  def __float__(self):
    if self.values[0]:
      return float(self.values[1])
    else:
      return "Error: Node has not been calculated yet"
    
def newWt(layers):
  res = []
  int = 0
  for i in range(len(layers)):
    if i == 0:
      int += 0
    else:
      int += layers[i] * layers[i-1]
  for i in range(int):
    res.append(1)
  return res

class Network:
  def __init__(self, layerList, UUID = None):
    self.UUID = UUID
    self.numl = layerList
    self.started = False
    self.refreshNodes()
    if UUID is None:
      self.UUID = Uuid(10)
      self.w = newWt(self.numl)
      self.saveWeights()
    else:
      self.UUID = UUID
      self.w = self.getWeights()

  def startWInt(self,testval = None):
    layerList = self.numl
    if testval is not None:
      layerList = testval
    self.WInt = []
    for i in range(len(layerList)):
      if i == 0:
        continue
      else:
        self.WInt.append(layerList[i - 1]*layerList[i])
    return self.WInt
  
  
  
  def refreshNodes(self):
    layerList = self.numl
    self.layers = []
    for i in range(len(layerList)):
      self.layers.append([])
      i = i
    for i in range(len(layerList)):
      for j in range(layerList[i]):
        j = j
        if i == 0:
          self.layers[i].append(Node(1))
        else:
          self.layers[i].append(Node(layerList[i-1]))
    self.startWInt()
  
  def start(self , inputList):
    if len(inputList) != len(self.layers[0]):
      print("Error: Input list does not match the amount of input nodes")
      return
    self.started = True
    currentWIndex = 0
    for i in range(len(self.layers[0])):
      self.layers[0][i].add(inputList[i], 1)
    
    for i in range(len(self.layers) - 1):
      for j in range(len(self.layers[i + 1])):
        for k in range(len(self.layers[i])):
          self.layers[i + 1][j].add(float(self.layers[i][k]), self.w[currentWIndex])
          currentWIndex = currentWIndex + 1
    return self.layers[-1]
  
  def __str__(self):
    if not self.started:
      return "Error: Network has not been started yet"
    else:
      res = ""
      y = self.layers[-1]
      for i in range(len(y)):
        res += f" [{y[i]}] "
      return res
    
  def getWeights(self):
    with open(f"weights{self.UUID}.json", "r") as f:
      return json.loads(f.read())[self.UUID]
  def saveWeights(self):
    with open(f"weights{self.UUID}.json","w") as f:
      json.dump({self.UUID:self.w},f)
    
  

def flatten(listToFlatten):
  res = []
  for i in listToFlatten:
    if isinstance(i, list):
      for j in i:
        res.append(j)
    else:
      res.append(i)
  return res    

def toFloat(layer):
  res = []
  for i in layer:
    res.append(float(i))
  return res

def getRelatedWeightIndexes(layerIndex, node, network, direction, Limit = 0):
  res = []
  int = 0
  for i in range(len(network.WInt)):
    if i == layerIndex - 1:
      break
    else:
      int += network.WInt[i]
  for i in range(len(network.layers[layerIndex])):
    if network.layers[layerIndex][i].uuid == node.uuid:
      nodeIndex = i
  prevLayer = network.layers[layerIndex-1]
  prevLayerFloat = []
  for i in range(len(prevLayer)):
    prevLayerFloat.append(float(prevLayer[i]))
  clw = []
  for i in range(len(prevLayerFloat)):
    try:
      clw.append(int + (nodeIndex * len(prevLayerFloat)) + i)
    except UnboundLocalError:
      return res
  clw = clw[::-1]
  clw = [clw[prevLayerFloat.index(direction(prevLayerFloat))]]
  if layerIndex - 1 >= 1 + Limit:
    clw.append(getRelatedWeightIndexes(layerIndex-1,network.layers[layerIndex-1][prevLayerFloat.index(direction(prevLayerFloat))],network,direction))
  else:
    return clw
  return flatten(clw)


def averageOfList(list):
  int = 0
  for i in range(len(list)):
    int += list[i]
  int = int/len(list)
  return int


def train(x, data, learning_rate = 0.01, epochs = 1, lim = 0, ignoreError = False):
  with open(data, "r") as f:
    data = json.loads(f.read())
  epochSuccessList = []
  corW = []
  print("\n----------------------------------------------------------------")
  print("\nSTART OF TRAINING" + f"\tNetwork: {[x]}\n")
  for i in range(epochs):
    success = []
    for j in range(len(data)):
      x.refreshNodes()
      outputLayer = x.start(data["input"][j])
      expectedRes = data["output"][j]
      outputLayerAsFloat = []
      for k in range(len(outputLayer)):
        outputLayerAsFloat.append(float(outputLayer[k]))
      for k in range(len(outputLayerAsFloat)):
        changes = {"up":[], "down": []}
        if outputLayerAsFloat[k] > expectedRes[k]:
          changes["down"].append(getRelatedWeightIndexes(x.layers.index(outputLayer),outputLayer[k],x,max,Limit=lim))
        elif outputLayerAsFloat[k] < expectedRes[k]:
          changes["up"].append(getRelatedWeightIndexes(x.layers.index(outputLayer),outputLayer[k],x,min, Limit=lim))
        error = abs(((outputLayerAsFloat[k] - expectedRes[k])/expectedRes[k]))
        success.append(abs(100 * ((outputLayerAsFloat[k] - expectedRes[k])/expectedRes[k])))
        for ul in changes['up']:
          for m in ul:
            if ignoreError:
              x.w[m] = x.w[m] + (learning_rate )
            else:
              x.w[m] = x.w[m] + (learning_rate * error)
        for ul in changes['down']:
          for m in ul:
            if ignoreError:
              x.w[m] = x.w[m] - (learning_rate )
            else:
              x.w[m] = x.w[m] - (learning_rate * error)
    print(f"Epoch #{i+1} Error Percentage: "+str(round(averageOfList(success),4))+"%")  
    epochSuccessList.append(averageOfList(success))
    corW.append(x.w)
  index = epochSuccessList.index(min(epochSuccessList))
  print(f"\n\nBest Epoch: #{index+1}" + f"\t\tError: {round(min(epochSuccessList),4)}%")
  print("Editing Weights")
  x.w = corW[index]
  print("Saving Weights")
  x.saveWeights()
  print("\nEND OF TRAINING\n")
  print("----------------------------------------------------------------")
  
    


def test(x, data, rep = 1):
  print("\n----------------------------------------------------------------\n")
  print("\nTESTING STARTED\n")
  for i in range(rep):
    for j in range(len(data["input"])):
      x.refreshNodes()
      x.start(data["input"][j])
      print("Output: " + str(toFloat(x.layers[-1])) + "\tExpected Result: " + str(data["output"][j]))
      print("\n")
  print("\nTESTING ENDED\n")
  print("\n----------------------------------------------------------------\n")
  
      






