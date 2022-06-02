import psutil
import os

def judgeprocess(processname):
    pl = psutil.pids()
    for pid in pl:
        if psutil.Process(pid).name() == processname:
            return True
    else:
        return False
        
if judgeprocess('CarlaUE4.exe') == False:
    os.startfile('D:\CARLA_0.9.11\WindowsNoEditor\CarlaUE4.exe')