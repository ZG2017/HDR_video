import PySpin
import cv2
import numpy as np
from run_CCRF_RGB import parallel_CCRF_B_4, parallel_CCRF_G_4,parallel_CCRF_R_4,parallel_CCRF_B_8, parallel_CCRF_G_8,parallel_CCRF_R_8,\
parallel_CCRF_B_2, parallel_CCRF_G_2,parallel_CCRF_R_2,parallel_CCRF_B_16, parallel_CCRF_G_16,parallel_CCRF_R_16
#parallel_CCRF_B_8, parallel_CCRF_G_8,parallel_CCRF_R_8, parallel_CCRF_B_16, parallel_CCRF_G_16,parallel_CCRF_R_16
from run_CCRF_mono import ldr_tonemap_rgb_image_cv2
from multiprocessing import Process,Queue,Value
import sys
import queue
from time import time
from math import *



class TriggerType:
    SOFTWARE = 1
    HARDWARE = 2




FRAME_WIDTH=400
FRAME_HEIGHT=300
BASE_EXPO=2000
db2 = 10*log(2)
db4 = 10*log(4)
db8 = 10*log(8)
db16= 10*log(16)
db32= 10*log(32)
db64= 10*log(64)
CHOSEN_TRIGGER = TriggerType.SOFTWARE


def print_instructions():
    print("Instructions: 'w' and 's' to modify k       'a' and 'd' to modify base exposure")
    print("              'i' and 'k' to modify power   'j' and 'l' to modify radius")
    print("              'f' to show/hide four frames  'm' to maximize HDR frame")
    print("              'q' to quit program           'h' to reshow instructions")
    print("              'e' to show/hide edge detection")
    print("              'p' to change CCRF calculation mode")

def CCRF_Lookup(f2q,fq, kValue):
    result_HDR = np.zeros(f2q.shape,np.uint8)
    if kValue == 2:
        result_HDR[:,:,0] = parallel_CCRF_B_2(f2q[:,:,0],fq[:,:,0])
        result_HDR[:,:,1] = parallel_CCRF_G_2(f2q[:,:,1],fq[:,:,1])
        result_HDR[:,:,2] = parallel_CCRF_R_2(f2q[:,:,2],fq[:,:,2])
    elif kValue == 4:
        result_HDR[:,:,0] = parallel_CCRF_B_4(f2q[:,:,0],fq[:,:,0])
        result_HDR[:,:,1] = parallel_CCRF_G_4(f2q[:,:,1],fq[:,:,1])
        result_HDR[:,:,2] = parallel_CCRF_R_4(f2q[:,:,2],fq[:,:,2])
    elif kValue == 8:
        result_HDR[:,:,0] = parallel_CCRF_B_8(f2q[:,:,0],fq[:,:,0])
        result_HDR[:,:,1] = parallel_CCRF_G_8(f2q[:,:,1],fq[:,:,1])
        result_HDR[:,:,2] = parallel_CCRF_R_8(f2q[:,:,2],fq[:,:,2])
    elif kValue == 16:
        result_HDR[:,:,0] = parallel_CCRF_B_16(f2q[:,:,0],fq[:,:,0])
        result_HDR[:,:,1] = parallel_CCRF_G_16(f2q[:,:,1],fq[:,:,1])
        result_HDR[:,:,2] = parallel_CCRF_R_16(f2q[:,:,2],fq[:,:,2])
    '''
    if kValue == 2:
        result_HDR[:,:,0] = parallel_CCRF_B_2(f2q[:,:,0],fq[:,:,0])
        result_HDR[:,:,1] = parallel_CCRF_G_2(f2q[:,:,1],fq[:,:,1])
        result_HDR[:,:,2] = parallel_CCRF_R_2(f2q[:,:,2],fq[:,:,2])
    
    elif kValue == 4:
        result_HDR[:,:,0] = parallel_CCRF_B_4(f2q[:,:,0],fq[:,:,0])
        result_HDR[:,:,1] = parallel_CCRF_G_4(f2q[:,:,1],fq[:,:,1])
        result_HDR[:,:,2] = parallel_CCRF_R_4(f2q[:,:,2],fq[:,:,2])
    elif kValue == 8:
        result_HDR[:,:,0] = parallel_CCRF_B_8(f2q[:,:,0],fq[:,:,0])
        result_HDR[:,:,1] = parallel_CCRF_G_8(f2q[:,:,1],fq[:,:,1])
        result_HDR[:,:,2] = parallel_CCRF_R_8(f2q[:,:,2],fq[:,:,2])
    elif kValue == 16:
        result_HDR[:,:,0] = parallel_CCRF_B_16(f2q[:,:,0],fq[:,:,0])
        result_HDR[:,:,1] = parallel_CCRF_G_16(f2q[:,:,1],fq[:,:,1])
        result_HDR[:,:,2] = parallel_CCRF_R_16(f2q[:,:,2],fq[:,:,2])'''
    return result_HDR

def base_exposure(cam,k):
    return 1/(cam.AcquisitionFrameRate.GetMax()/8)/sum([(k**i) for i in range(0,4)])*1000000 #10000000micros

def configure_s(cam):
    try:
        cam.GammaEnable.SetValue(False)
        cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)
        cam.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Red)
        cam.BalanceRatio.SetValue(1.1) 
        cam.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Blue)
        cam.BalanceRatio.SetValue(1.8) 
        #cam.AutoExposureEVCompensation.SetValue(False) 
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        cam.GainAuto.SetValue(False)
        cam.Gain.SetValue(0)
        cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit8)
        cam.AutoExposureTargetGreyValueAuto.SetValue(PySpin.AutoExposureTargetGreyValueAuto_Off)
        
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(cam.AcquisitionFrameRate.GetMax())
        cam.AcquisitionFrameRateEnable.SetValue(False)
        print("Maximum achievable framerate is : ", cam.AcquisitionFrameRate.GetMax())
        return True
    except:
        print("some error occured in settings. \n Maximum achievable framerate is : ",cam.AcquisitionFrameRate.GetMax())
        return False

def configure_g(cam,db):
    try:
        result = True
        cam.Gain.SetValue(db)
        return result
    except:
        result = False
        print("error")
        return result

def configure_t(cam):

    if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
        print("Software trigger chosen...")
    elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
        print("Hardware trigger chose...")

    try:
        result = True

        if cam.TriggerMode.GetAccessMode() != PySpin.RW:
            print("Unable to disable trigger mode (node retrieval). Aborting...")
            return False

        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        
        print("Trigger mode disabled...")

        if cam.TriggerSource.GetAccessMode() != PySpin.RW:
            print("Unable to get trigger source (node retrieval). Aborting...")
            return False

        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)

        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        
        print("Trigger mode turned back on...")

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        return False

    return result

def grab_next_image_by_trigger(cam):
    """
    This function acquires an image by executing the trigger node.

    :param cam: Camera to acquire images from.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            if cam.TriggerSoftware.GetAccessMode() != PySpin.WO:
                print("Unable to execute trigger. Aborting...")

            cam.TriggerSoftware.Execute()

            # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger

        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            print("Use the hardware to trigger image acquisition.")

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        #return False


def acquire_images(cam, nodemap):
    """
    This function acquires 10 images from a device, stores them in a list, and returns the list.
    please see the Acquisition example for more in-depth comments on acquiring images.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    #print("*** IMAGE ACQUISITION ***\n")
    try:
        result = True
        try:
            grab_next_image_by_trigger(cam)

            image_result = cam.GetNextImage()

            if image_result.IsIncomplete():
                print("Image incomplete with image status %d..." % image_result.GetImageStatus())

            else:
                w = image_result.GetWidth()
                h = image_result.GetHeight()

                image_converted = image_result.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)

                image_result.Release()
                #print("")

        except PySpin.SpinnakerException as ex:
            print("Error: %s" % ex)
            result = False

        #cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        result = False

    return result, image_converted,w,h

def reset_trigger(cam):
    """
    This function returns the camera to a normal state by turning off trigger mode.

    :param cam: Camera to acquire images from.
    :type cam: CameraPtr
    :returns: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        if cam.TriggerMode.GetAccessMode() != PySpin.RW:
            print("Unable to disable trigger mode (node retrieval). Aborting...")
            return False

        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

        print("Trigger mode disabled...")

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        result = False

    return result

def run_single_camera(cam,to_proc,sharedK,sharedBaseExposure):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run example on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    try:
        cam.Init()
        
        configure_s(cam)

        nodemap = cam.GetNodeMap()
        num=0

        if configure_t(cam) is False:
            return False

        if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
            print("Unable to disable automatic exposure. Aborting...")
            return False
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print("Unable to set acquisition mode to continuous (enum retrieval). Aborting...")
            return False

        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(node_acquisition_mode_continuous):
            print("Unable to set acquisition mode to continuous (entry retrieval). Aborting...")
            return False

        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
        
        print("Acquisition mode set to continuous...")

        print("Automatic exposure disabled...")

        canvas=np.zeros((FRAME_HEIGHT*2,FRAME_WIDTH*2,3), np.uint8)
        cam.BeginAcquisition()
        err, img,w,h = acquire_images(cam, nodemap)
        if err < 0:
            return err
        half_fh = int(FRAME_HEIGHT/2)
        half_fw = int(FRAME_WIDTH/2)
        half_h = int(h/2)
        half_w = int(w/2)

        bot = half_h-half_fh
        top = half_h+half_fh
        left = half_w-half_fw
        right = half_w+half_fw
        HDR_FRAME = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3))
        
        while True:
            exps=[(sharedK.value**i)*sharedBaseExposure.value for i in range(0,6)]
            #exps[1] = exps[1]*5
            #exps[2] = exps[2]*5
            #exps[3] = exps[3]*5 
            configure_exposure(cam, exps[num])
            if num not in [4,5]:
                configure_g(cam,0)
            if num == 4:
                configure_g(cam,db2)
            elif num ==5:
                configure_g(cam,db4)
            else:
                configure_exposure(cam, exps[num])

            err, img,w,h = acquire_images(cam, nodemap)

            img = img.GetData().reshape(h,w,3)
            
            img = img[bot:top,left:right]
            to_proc.put((img,num))
            num+=1
            if num>=len(exps):
                num=0
            #print(p.is_alive())
            if p.is_alive() is False:
                try:
                    print("Stopping image acquisition")
                    break
                except:
                    print("Error occured in stopping image acquisition")
        cv2.destroyAllWindows()
        cam.EndAcquisition()
        reset_trigger(cam)
        cam.DeInit()
        print("Successfully emptied main process, exiting run_single_camera")

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        result = False


def configure_exposure(cam,exposure):
    #print("*** CONFIGURING EXPOSURE ***\n")

    try:
        result = True

        if cam.ExposureTime.GetAccessMode() != PySpin.RW:
            print("Unable to set exposure time. Aborting...")
            return False

        cam.ExposureTime.SetValue(exposure)

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        result = False

    return result

def h_proc(pipe, sharedK, sharedBaseExposure):
    frame1 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    frame2 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    frame3 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    frame4 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    #6 frames
    frame5 = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    frame6 = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)

    layer1_1 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer1_2 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer1_3 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer1_4 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer1_5 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer2_1 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer2_2 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer2_3 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer2_4 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer3_1=np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer3_2=np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer3_3=np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer4_1=np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer4_2=np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer5_1=np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer1_1_pair = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer1_2_pair = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer1_3_pair = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer2_1_pair = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer2_2_pair = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    layer3_1_pair = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    edges = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)

    calc_temp = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    result_HDR =np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    HDR_FRAME = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3),np.uint8)
    debug_flag = True
    max_flag = False
    CCRF_mode_flag = True
    edge_flag = False
    canvas=np.zeros((FRAME_HEIGHT*2,FRAME_WIDTH*3,3), np.uint8)
    #intermediate_canvas=np.zeros((FRAME_HEIGHT*2,FRAME_WIDTH*3,3), np.uint8)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', ((canvas[:,:,0].shape)[1],(canvas[:,:,0].shape)[0]))
    cv2.startWindowThread()
    cv2.namedWindow('HDR', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('HDR', ((frame1[:,:,0].shape)[1],(frame1[:,:,0].shape)[0]))
    cv2.startWindowThread()
    #cv2.namedWindow('Intermediate calculations', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Intermediate calculations', ((intermediate_canvas[:,:,0].shape)[1],(intermediate_canvas[:,:,0].shape)[0]))
    #cv2.startWindowThread()
    power = 8
    radius = 30
    print_instructions()
    while True:
        try:
            v=pipe.get()
            if v is None:
                break
            if debug_flag == True:
                canvas[0:FRAME_HEIGHT,0:FRAME_WIDTH] = frame1
                canvas[0:FRAME_HEIGHT,FRAME_WIDTH:FRAME_WIDTH*2] = frame2
                canvas[0:FRAME_HEIGHT,FRAME_WIDTH*2:FRAME_WIDTH*3]=frame3
                canvas[FRAME_HEIGHT:FRAME_HEIGHT*2,0:FRAME_WIDTH] =frame4
                canvas[FRAME_HEIGHT:FRAME_HEIGHT*2,FRAME_WIDTH:FRAME_WIDTH*2] = frame5
                canvas[FRAME_HEIGHT:FRAME_HEIGHT*2,FRAME_WIDTH*2:FRAME_WIDTH*3] = frame6
                cv2.imshow("frame",canvas)
                #intermediate_canvas[0:FRAME_HEIGHT,0:FRAME_WIDTH] = calc_temp
                #intermediate_canvas[0:FRAME_HEIGHT,FRAME_WIDTH:FRAME_WIDTH*2] = calc2
                #intermediate_canvas[0:FRAME_HEIGHT,FRAME_WIDTH*2:FRAME_WIDTH*3] = calc3
                #intermediate_canvas[FRAME_HEIGHT:FRAME_HEIGHT*2,0:FRAME_WIDTH]=calc4
                #intermediate_canvas[FRAME_HEIGHT:FRAME_HEIGHT*2,FRAME_WIDTH:FRAME_WIDTH*2]=calc5
                #intermediate_canvas[FRAME_HEIGHT:FRAME_HEIGHT*2,FRAME_WIDTH*2:FRAME_WIDTH*3]=result_HDR
                #cv2.imshow("Intermediate calculations",intermediate_canvas)
            if v[1] == 1:
                if CCRF_mode_flag == True:
                    layer1_1 = CCRF_Lookup(v[0],frame1,sharedK.value)
                else:
                    layer1_1_pair = CCRF_Lookup(v[0],frame1,sharedK.value**2)
                #calc_temp = ldr_tonemap_rgb_image_cv2(calc1/255,power ,radius)
                #calc_temp = calc_temp*255
                frame2 = v[0]
            elif v[1] ==2:
                layer1_2 = CCRF_Lookup(v[0],frame2,sharedK.value)
                layer2_1 = CCRF_Lookup(layer1_2,layer1_1,sharedK.value)
                frame3 = v[0]
            elif v[1] ==3:
                if CCRF_mode_flag == True:
                    layer1_3 = CCRF_Lookup(v[0],frame3,sharedK.value)
                    layer2_2 = CCRF_Lookup(layer1_3,layer1_2,sharedK.value)
                    layer3_1 = CCRF_Lookup(layer2_2,layer2_1,sharedK.value)
                else:
                    layer1_2_pair = CCRF_Lookup(v[0],frame3,sharedK.value**2)
                    layer2_1_pair = CCRF_Lookup(layer1_2_pair,layer1_1_pair,sharedK.value**2)
                frame4 = v[0]
            elif v[1] ==4:
                layer1_4 = CCRF_Lookup(v[0],frame4,sharedK.value)
                layer2_3 = CCRF_Lookup(layer1_4,layer1_3,sharedK.value)
                layer3_2 = CCRF_Lookup(layer2_3,layer2_2,sharedK.value)
                layer4_1 = CCRF_Lookup(layer3_2,layer3_1,sharedK.value)
                frame5 = v[0]

            elif v[1] ==5: #v[1] == f8q
                if CCRF_mode_flag == True:
                    layer1_5 = CCRF_Lookup(v[0],frame5,sharedK.value)
                    layer2_4 = CCRF_Lookup(layer1_5,layer1_4,sharedK.value)
                    layer3_3 = CCRF_Lookup(layer2_4,layer2_3,sharedK.value)
                    layer4_2 = CCRF_Lookup(layer3_3,layer3_2,sharedK.value)
                    layer5_1 = CCRF_Lookup(layer4_2,layer4_1,sharedK.value)
                    HDR_FRAME = ldr_tonemap_rgb_image_cv2(layer5_1/255,power ,radius)
                    
                else:
                    layer1_3_pair = CCRF_Lookup(v[0],frame5,sharedK.value**2)
                    layer2_2_pair = CCRF_Lookup(layer1_3_pair,layer1_2_pair,sharedK.value**2)
                    layer3_1_pair = CCRF_Lookup(layer2_2_pair,layer2_1_pair,sharedK.value**2)
                    HDR_FRAME = ldr_tonemap_rgb_image_cv2(layer3_1_pair/255,power ,radius**2)
                
                if edge_flag == True:
                    edges = cv2.Canny(np.uint8(HDR_FRAME*255),200,200)
                    cv2.imshow("edge_detection",edges)
                #print("Time taken to spit one HDR frame = ",time()-init_time)
                #init_time = time()
                frame6 = v[0]
                #result_HDR = HDR_FRAME*255
                cv2.imshow("HDR",HDR_FRAME)
                #intermediate_canvas[FRAME_HEIGHT:FRAME_HEIGHT*2,FRAME_WIDTH*2:FRAME_WIDTH*3]=HDR_FRAME
                #if debug_flag == False:
                #    cv2.imshow("HDR",HDR_FRAME)
                userInput =  cv2.waitKey(1)
                if userInput & 0xff == ord('w'):
                    if CCRF_mode_flag == False:
                        if sharedK.value == 4:
                            print("Doing less pair calculation mode, cannot go over k=4")
                            #sharedK.value *= 2
                            #print("Modified k to", sharedK.value)
                        else:
                            sharedK.value *= 2
                            print("Modified k to", sharedK.value)
                    else:
                        if sharedK.value != 16:
                            sharedK.value *= 2
                            print("Modified k to", sharedK.value)
                        else:
                            print("Warning: Maximum k is 16")
                elif userInput & 0xff == ord('s'):
                    if sharedK.value != 2:
                        sharedK.value = int(sharedK.value / 2)
                        print("Modified k to", sharedK.value)
                    else:
                        print("Warning: Minimum k is 2")
                elif userInput & 0xff == ord('d'):
                    if sharedBaseExposure.value < 4096:
                        sharedBaseExposure.value *= 2
                        print("Modified BaseExposure to {} microseconds".format(sharedBaseExposure.value))
                    else:
                        print("Warning: Maximum BaseExposure is 4096 microseconds")
                elif userInput & 0xff == ord('a'):
                    if sharedBaseExposure.value >4:
                        sharedBaseExposure.value /= 2
                        print("Modified BaseExposure to {} microseconds".format(sharedBaseExposure.value))
                    else:
                        print("Warning: Minimum BaseExposure is 4 microseconds")
                elif userInput & 0xff==ord('q'):
                    break
                elif userInput & 0xff==ord('f'):
                    if debug_flag == True:
                        cv2.destroyWindow('frame')
                        #cv2.destroyWindow('Intermediate calculations')
                        cv2.waitKey(1)
                        #cv2.namedWindow('HDR', cv2.WINDOW_NORMAL)
                        #cv2.resizeWindow('HDR', ((frame1[:,:,0].shape)[1],(frame1[:,:,0].shape)[0]))
                        #cv2.startWindowThread()
                    else:
                        #cv2.destroyWindow('HDR')
                        cv2.waitKey(1)
                    debug_flag = not debug_flag
                elif userInput & 0xff==ord('m'):
                    if max_flag == False:
                        #cv2.resizeWindow("HDR",cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty("HDR",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                        cv2.waitKey(1)
                    if max_flag == True:
                        #cv2.resizeWindow("HDR",frame1[:,:,0].shape)
                        cv2.setWindowProperty("HDR",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
                        cv2.waitKey(1)
                    max_flag = not max_flag
                    #print(max_flag)
                elif userInput & 0xff==ord('i'):
                    power += 1
                    print("Modified power to",power)
                    if power == 20:
                        print("Yes, there is no power limit, but who knows what will happen")
                elif userInput & 0xff==ord('k'):
                    power -= 1
                    print("Modified power to",power)
                elif userInput & 0xff==ord('l'):
                    radius += 5
                    print("Modified radius to",radius)
                elif userInput & 0xff==ord('j'):
                    radius -= 5
                    print("Modified radius to",radius)
                elif userInput & 0xff==ord('h'):
                    print_instructions()
                elif userInput & 0xff==ord('p'):
                    if CCRF_mode_flag == True:
                        if sharedK.value not in [2,4]:
                            print("K value is higher than 4, no usable LUT")
                        else:
                            print("Changed CCRF calculation to 3 calculations")
                            CCRF_mode_flag = not CCRF_mode_flag
                    else:
                        print("Changed CCRF calculation to 6 calculations")
                        CCRF_mode_flag = not CCRF_mode_flag
                elif userInput & 0xff==ord('e'):
                    if edge_flag == False:
                        cv2.namedWindow('edge_detection', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('edge_detection', ((frame1[:,:,0].shape)[1],(frame1[:,:,0].shape)[0]))
                        cv2.startWindowThread()
                        cv2.waitKey(1)
                        edge_flag = not edge_flag
                    else:
                        cv2.destroyWindow('edge_detection')
                        cv2.waitKey(1)
                        edge_flag = not edge_flag

                    


            elif v[1] == 0:
                frame1 = v[0]

            #print(result)
        except queue.Empty:
            continue
        except Exception as e:
            raise e
    cv2.destroyAllWindows()
            

if __name__ == "__main__":
    sharedK = Value('i', 2)
    sharedBaseExposure = Value('d',64)
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print("Number of cameras detected:", num_cameras)
    if num_cameras == 0:
        cam_list.Clear()

        system.ReleaseInstance()

        print("Not enough cameras!")
        sys.exit(0)
    
    to_proc = Queue()
    p = Process(target=h_proc, args=(to_proc,sharedK, sharedBaseExposure))
    p.start()

    cam = cam_list.GetByIndex(0)
    run_single_camera(cam,to_proc,sharedK,sharedBaseExposure)
    #-----------------------------------------------
    print("Successfully exit run_single_camera")
    print("Number of stacked frames that have not been used: ",to_proc.qsize())
    clear_count = 0

    while to_proc.empty() is False:
        #print("in loop")
        clear_count += 1
        try:
            #print("Frame{}:".format(clear_count),to_proc.get()[0][0][0])
            #print(to_proc.get())
            to_proc.get()
        except:
            continue
    p.join()
    #-----------------------------------------
    del cam

    cam_list.Clear()

    system.ReleaseInstance()

    print("Reached end of main")
    print("Status of child process: dead" if p.is_alive() is False else "Status of child process: alive")
    while p.is_alive():
        print("wait")
        continue
    to_proc.close()