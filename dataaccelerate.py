#pip install prefetch_generator

# 新建DataLoaderX类
from torch.utils.data import DataLoader
import numpy as np
import torch

def sendall2gpu(listinlist,device):
    if isinstance(listinlist,(list,tuple)):
        return [sendall2gpu(_list,device) for _list in listinlist]
    elif isinstance(listinlist, (dict)):
        return dict([(key,sendall2gpu(val,device)) for key,val in listinlist.items()])
    elif isinstance(listinlist, np.ndarray):
        return torch.from_numpy(listinlist).to(device=device, non_blocking=True)
    else:
        return listinlist.to(device=device, non_blocking=True)
try:
    from prefetch_generator import BackgroundGenerator
    class DataLoaderX(DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())
except:
    pass#DataLoaderX = DataLoader
class DataSimfetcher():
    def __init__(self, loader, device='auto'):
    
        if device == 'auto':
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.loader = iter(loader)

    def next(self):
        try:

            self.batch = next(self.loader)
            self.batch = sendall2gpu(self.batch,self.device)
        except StopIteration:
            self.batch = None
        return self.batch
class DataPrefetcher():
    def __init__(self, loader, device='auto'):
        if device == 'auto':
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device#raise NotImplementedError
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = sendall2gpu(self.batch,self.device)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

class infinite_batcher:
    def __init__(self,data_loader, device='auto'):
        self.length=len(data_loader)
        self.now=-1
        self.data_loader= data_loader
        self.prefetcher = None
        self.device     = device
    def next(self):
        if (self.now >= self.length) or (self.now == -1):
            if self.prefetcher is not None:del self.prefetcher
            self.prefetcher = DataSimfetcher(self.data_loader,device=self.device)
            self.now=0
        self.now+=1
        return self.prefetcher.next()

