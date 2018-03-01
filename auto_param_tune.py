import os
import sys
import threading
import time
import subprocess
import signal

home_folder = os.path.expanduser("~")
user_site_packages_folder = "{}/.local/lib/python2.7/site-packages".format(home_folder)
if user_site_packages_folder not in sys.path:
    sys.path.append(user_site_packages_folder)
import pynvml as nv
#import pynvml_local as nv 
import psutil

def kill_process(pid):
    kill = signal.SIGKILL
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for p in children:
        p.send_signal(kill)
    #pid.send_signal(kill)
    

def GPU_exception_734(available_list):
    output = []
    for x in available_list:
        if(x != "1"):
            output.append(x)
    return output


def AskGPU():
    available_list = GetAvailableGPUList()
    if len(available_list) > 0:
        #available_list = exception_adjust(available_list)
        available_list = tuple(available_list)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        return

    gpuidx = ','.join(available_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuidx
    
    return

def GetAvailableGPUList():
    deviceCount = nv.nvmlDeviceGetCount()
    output = []
    for i in range(deviceCount):
        
        #this is only for nasic server
        #i = ((i-1)%3)
        
        handle = nv.nvmlDeviceGetHandleByIndex(i)
        if(IsGPUAvailable(handle)):
            output.append(str(i))
    
    #output = GPU_exception_734(output)
    return output

def IsGPUAvailable(handle):
    meminfo = nv.nvmlDeviceGetMemoryInfo(handle)
    used = meminfo.used * 1.0
    total = meminfo.total * 1.0
    if (used / total) < 0.5:
        return True
    else:
        return False
    return False

def miner(mine_sleep, IsStop):
    tname = "[MINE]"
    #miner_command = "sh ~/zero-convnet/train-zec.sh"
    mine_controller = subprocess.Popen(args = ('sh', '/home/NASICLab/nmsoc1/zero-convnet/train-zec.sh'))
    #mine_controller = subprocess.Popen(args = ('sh', '/home/remote-test/zcash-miner/train-zec.sh'))
    #mine_controller = subprocess.Popen(args = ('sh', '/home/kuan/zcash-miner/train-zec.sh'))
                                       #stdin=None, stdout=None, stderr=None, close_fds=True)
    print(tname + "miner starts working")
    print(tname + "miner status is " + str(mine_controller.poll()))
    #subprocess.call("export | grep CUDA", shell = True, )
    #print("IsStop flag is ", IsStop.isSet())
    while(not IsStop.isSet()):
        time.sleep(60 * mine_sleep)
        print(tname + "Check if there is a stop event")
        print(tname + "mine_controller's status is " +  mine_controller.poll())
        if(not mine_controller.poll() == None):
            print(tname + "The miner process does not start properly")
            try:
                kill_process(mine_controller.pid)
                return
            except:
                return
    print(tname + "get stop signal")
    kill_process(mine_controller.pid)
    print(tname + "stop the miner because someone is using GPU")
    return

def Mine_Thread(IsStop, mine_sleep):
    mine_thread = threading.Thread(target = miner, name = 'miner', args = (mine_sleep, IsStop))
    #mine_thread.setDaemon(True)
    
    return mine_thread 

def monitor():
    tname = "[MONITOR]"
    IsStop = threading.Event()
    mine_sleep = 1 #in minute
    monitor_sleep = 1 #in minute
    while(True):
        current_gpu = os.getenv("CUDA_VISIBLE_DEVICES").split(',')
        available_gpu = GetAvailableGPUList()
        print(tname + "current gpu is " + ','.join(current_gpu)) 
        print(tname + "available gpu is " + ','.join(available_gpu)) 
        if(len(available_gpu) != 0):
            print(tname + "there are some available gpus")
            if(current_gpu == available_gpu):
                print(tname + "no gpu allocation change")
                time.sleep(60 * monitor_sleep)                     
            else:
                print(tname + "set stopevent is TRUE")
                IsStop.set()
                print(tname + "reset the miner and wait for restarting it")
                time.sleep(60*(mine_sleep*1.2))
                IsStop.clear()
                AskGPU()
                print(tname + "now, allocated gpu id:" + " ".join(str(x) for x in os.getenv("CUDA_VISIBLE_DEVICES")))
                print(tname + "start a miner thread")
                mine_thread = Mine_Thread(IsStop, mine_sleep)
                mine_thread.start()
                
        elif(len(available_gpu) == 0):
            print(tname + "no available gpu")
            print(tname + "set stopevent is TRUE")
            IsStop.set()
        time.sleep(60 * monitor_sleep)
        #print(tname + "mine thread alive?" + str(mine_thread.isAlive()))
        #print(threading.activeCount())
	if(threading.activeCount() == 2):
	    print(tname + "miner thread is not activated currently")
            print(tname + "set CUDA_VISIBLE_DEVICES to default and return")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""            

def Monitor_Thread():
    monitor_thread = threading.Thread(target = monitor, name = 'monitor')
    #monitor_thread.setDaemon(True)
    return monitor_thread


nv.nvmlInit()

monitor_thread = Monitor_Thread()
monitor_thread.start()
monitor_thread.join()

nv.nvmlShutdown()

