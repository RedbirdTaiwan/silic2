# -*- coding: utf-8 -*-
import argparse
import json
import mimetypes
import os
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib

# Spectrograms are rendered to NumPy arrays, never to an interactive window.
# Using a GUI backend (for example TkAgg) from the UI worker thread causes
# Tk_GetPixmap failures on Windows.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from pydub import AudioSegment, effects, scipy_effects
from nnAudio import features
import scipy.signal as signal
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    # convert the sound with altered frame rate to a standard frame rate
    # so that regular playback programs will work right. They often only
    # know how to play audio at standard frame rate (like 44.1k)
    print(sound_with_altered_frame_rate.frame_rate)
    return sound_with_altered_frame_rate.set_frame_rate(int(sound.frame_rate*speed))

# 計算每個聲道的 RMS 音量
def calculate_rms(audio_segment):
    """ 計算音訊片段的 RMS 音量 """
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float64)  # 使用 float64 避免數值溢出

    # ✅ 檢查是否為空音訊
    if samples.size == 0:
        return 0.0  # 避免空音訊導致錯誤

    # ✅ 確保所有樣本為有限值，避免 NaN 計算
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

    # ✅ 計算 RMS，確保值不會變成 NaN 或負數
    mean_square = np.mean(samples**2)
    if mean_square < 0 or np.isnan(mean_square):  # 檢查非法值
        return 0.0

    rms_value = np.sqrt(mean_square)

    return rms_value

def AudioStandarize(audio_file, sr=32000, device=None, high_pass=0, ultrasonic=False):
  if not device:
      device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  filext = os.path.splitext(audio_file)[1].lower()[1:]
  try:
      sound = AudioSegment.from_file(audio_file)
  except Exception:
      try:
          if filext == "mp3":
              sound = AudioSegment.from_mp3(audio_file)
          elif filext == "ogg":
              sound = AudioSegment.from_ogg(audio_file)
          else:
              sound = AudioSegment.from_wav(audio_file)
      except Exception:
          print(f'Sorry, cannot open your file with extension {filext}.')
          return None
  original_metadata = {'channel': sound.channels, 'sample_rate':sound.frame_rate, 'sample_size':len(sound.get_array_of_samples()), 'duration':sound.duration_seconds}
  print('Origional audio: channel = %s, sample_rate = %s Hz, sample_size = %s, duration = %s s' %(original_metadata['channel'], original_metadata['sample_rate'], original_metadata['sample_size'], original_metadata['duration']))
  if ultrasonic:
      if sound.frame_rate > 100000: # UltraSonic
          sound = speed_change(sound, 1/12)
      else:
          return False
  if sound.frame_rate > sr:
      sound = scipy_effects.low_pass_filter(sound, sr/2)
  if sound.frame_rate != sr:
      sound = sound.set_frame_rate(sr)
  if sound.channels > 1:
      try:
        left_channel = sound.split_to_mono()[0]
        right_channel = sound.split_to_mono()[1]
        left_rms = calculate_rms(left_channel)
        right_rms = calculate_rms(right_channel)
        if left_rms >= right_rms:
          sound = left_channel
        else:
          sound = right_channel
      except Exception:
        sound = left_channel
  if not sound.sample_width == 2:
      sound = sound.set_sample_width(2)
  if high_pass:
    sound = sound.high_pass_filter(high_pass)
  sound = effects.normalize(sound) # normalize max-amplitude to 0 dB
  songdata = np.array(sound.get_array_of_samples())
  duration = round(songdata.shape[0]/sound.frame_rate*1000) #ms
  audiodata = torch.tensor(songdata, device=device).float()
  print('Standarized audio: channel = %s, sample_rate = %s Hz, sample_size = %s, duration = %s s' %(sound.channels, sound.frame_rate, songdata.shape[0], sound.duration_seconds))
  return sound.frame_rate, audiodata, duration, sound, original_metadata

def get_media_files(directory):
  if os.path.isfile(directory):
    mime_type, _ = mimetypes.guess_type(directory)
    if mime_type and (mime_type.startswith('audio') or mime_type.startswith('video')):
      return [os.path.basename(directory)]
    return []

  if not os.path.isdir(directory):
    return []

  media_files = []

  for filename in os.listdir(directory):
    # Get the full path of the file
    filepath = os.path.join(directory, filename)

    # Guess the MIME type of the file
    mime_type, _ = mimetypes.guess_type(filepath)

    if mime_type is not None:
      # If the MIME type is audio or video, add the filename to the list
      if mime_type.startswith('audio') or mime_type.startswith('video'):
        media_files.append(filename)

  return media_files

class Silic:
  """
    Arguments:
        sr (int): Sampling Rate in Hz
        n_fft (int): Window(Frame) Size in samples
        hop_length (str): Frame Step (or Hop Size) in samples
        n_mels (int): The number of Mel filter banks
        fmin (int): The starting frequency for the lowest Mel filter bank in Hz
        fmax (int): The ending frequency for the highest Mel filter bank in Hz
        clip_length (int): The duration of each inference in ms
  """
  def __init__(self, sr=32000, n_fft=1600, hop_length=400, n_mels=240, fmin=100, fmax=15000, device=None, clip_length=3000):
    self.sr = sr
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.n_mels = n_mels
    self.fmin = fmin
    self.fmax = fmax
    self.clip_length = clip_length
    if device:
      self.device = device
    else:
      self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    self.spec_layer = features.STFT(sr=sr, n_fft=n_fft, hop_length=hop_length).to(self.device)
    self.spec_mel_layer = features.MelSpectrogram(sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, window='hann', center=True, pad_mode='reflect', power=2.0, htk=False, fmin=fmin, fmax=fmax, norm=1, verbose=True).to(self.device)
    self.rainbow_img = torch.tensor([], dtype=torch.float32, device=self.device)
    self.model_path = None
    self.model = None
    self.names = None
    self.soundclasses = None
  
  def audio(self, audio_file, ultrasonic=False):
    self.audiofilename = os.path.basename(audio_file)
    self.audiofilename_without_ext = os.path.splitext(self.audiofilename)[0]
    self.audiopath = os.path.dirname(audio_file)
    self.audiofileext = audio_file.split('.')[-1]
    standardized = AudioStandarize(audio_file, self.sr, self.device, high_pass=self.fmin, ultrasonic=ultrasonic)
    if standardized is None or standardized is False:
      raise ValueError(f'Unable to standardize audio file: {audio_file}')
    self.sr, self.audiodata, self.duration, self.sound, self.original_metadata = standardized
    self.original_sound = AudioSegment.from_file(audio_file)
    self.analysis_audio = self.original_sound.split_to_mono()[0] if self.original_sound.channels > 1 else self.original_sound

  def save_standarized(self, targetmp3path=None):
    if not targetmp3path:
      targetmp3path = os.path.join(self.audiopath, 'mp3', '%s.mp3'%self.audiofilename_without_ext)
      if not os.path.isdir(os.path.dirname(targetmp3path)):
        os.makedirs(os.path.dirname(targetmp3path))
    self.sound.export(targetmp3path, bitrate="128k", format="mp3")
    print('Standarized audio was saved to %s' %targetmp3path)
    return targetmp3path
    
  def spectrogram(self, audiodata, spect_type='linear', rainbow_bands=5):
    if spect_type in ['mel', 'rainbow']:
      spec = self.spec_mel_layer(audiodata)
      w = spec.size()[2]/55
      h = spec.size()[1]/55
      if spect_type == 'mel':
        fig = plt.figure(figsize=(w, h), dpi=100)
        data = torch.sqrt(torch.sqrt(torch.abs(spec[0]) + 1e-6)).cpu().numpy()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.imshow(data, origin='lower', cmap='gray_r', aspect='auto')
      elif rainbow_bands > 1:
        fig, ax = plt.subplots(rainbow_bands, gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(w, h))
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        data = torch.log(torch.log(spec[0] + 1e-6))
        for i in range(rainbow_bands):
          subdata = data[i*int(self.n_mels/rainbow_bands):(i+1)*int(self.n_mels/rainbow_bands)].cpu().numpy()
          ax[rainbow_bands-i-1].set_axis_off()
          ax[rainbow_bands-i-1].pcolormesh(subdata, cmap=ListedColormap(cm.rainbow(np.linspace((i+1)/rainbow_bands, (i/rainbow_bands), 32))), rasterized=True)
      else:
        print('Bins of Rainbow should larger than 0.')
        return False
    else:
      spec = self.spec_layer(audiodata)
      data = torch.sqrt(torch.sqrt(torch.abs(spec[0]) + 1e-6)).cpu().numpy()[:,:,0]
      w = data.shape[1]/100*(5/4)*2
      h = data.shape[0]/100*(1/4)*2
      fig = plt.figure(figsize=(w, h), dpi=100)
      plt.gca().set_axis_off()
      plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
      plt.imshow(data, origin='lower', cmap='gray_r', aspect='auto')
    
    fig.canvas.draw()
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = fig.canvas.tostring_rgb()
        img = np.frombuffer(buf, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    elif hasattr(fig.canvas, "tostring_argb"):
        buf = fig.canvas.tostring_argb()
        argb = np.frombuffer(buf, dtype=np.uint8)
        argb = argb.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # ARGB → RGB
        img = argb[:, :, 1:4]
    else:
        raise RuntimeError("Canvas backend does not support tostring_rgb or tostring_argb")
    cv2_img = img #cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    
    return cv2_img

  def tfr(self, targetfilepath=None, spect_type='linear', rainbow_bands=5, start=0, stop=None):
    if self.clip_length and ((self.audiodata.size()[0] / self.sr * 1000) < self.clip_length):
        self.audiodata = torch.cat((self.audiodata, torch.zeros(round(self.clip_length*self.sr/1000)-self.audiodata.size()[0], device=self.device)), 0)
    if not stop:
        stop = self.duration
    max_sample_size = 1920000
    if not targetfilepath:
      targetfilepath = os.path.join(self.audiopath, spect_type, '%s.png'%self.audiofilename_without_ext)
      if not os.path.isdir(os.path.dirname(targetfilepath)):
        os.makedirs(os.path.dirname(targetfilepath))
    target_dir = os.path.dirname(targetfilepath) or '.'
    if not os.path.isdir(target_dir):
      raise FileNotFoundError(f'Cannot find the target folder {target_dir}.')
    if (stop - start)/1000*self.sr > (max_sample_size):
        if not os.path.exists('tmp'):
            try:
                os.makedirs('tmp')
            except OSError as exc:
                raise OSError('Cannot create tmp folder!') from exc
        
        imgs = []
        for ts in range(int(round(start/1000*self.sr)), int(round(stop/1000*self.sr)-self.sr*0.1), max_sample_size):
            if ts+max_sample_size > round(stop/1000*self.sr):
              data = self.audiodata[ts:round(stop/1000*self.sr)+1]
            else:
              data = self.audiodata[ts:ts+max_sample_size]
            try:
              imgs.append(self.spectrogram(data, spect_type, rainbow_bands=rainbow_bands))
            except Exception as exc:
              raise RuntimeError('Error while converting the spectrogram') from exc
        self.cv2_img = cv2.hconcat(imgs)
    else:
        self.cv2_img = self.spectrogram(self.audiodata[int(round(start/1000*self.sr)):int(round(stop/1000*self.sr))], spect_type, rainbow_bands=rainbow_bands)
    
    if spect_type == 'rainbow' and rainbow_bands == 5:
      self.rainbow_img = cv2.cvtColor(self.cv2_img, cv2.COLOR_RGB2BGR)
    
    height, width, colors = self.cv2_img.shape
    #cv2.imwrite(targetfilepath, self.cv2_img)
    PILimage = Image.fromarray(self.cv2_img)
    try:
      PILimage.save(targetfilepath, dpi=(72,72))
    except OSError:
      targetfilepath = '%spng' %targetfilepath[:-3]
      PILimage.save(targetfilepath, dpi=(72,72))
    print('Spectrogram was saved to %s.'%targetfilepath)
    return targetfilepath

  def mel_to_freq(self, mel):
    if mel < 0:
      return self.fmin
    mel = mel*(1127*np.log(1+self.fmax/700)-1127*np.log(1+self.fmin/700)) + 1127*np.log(1+self.fmin/700)
    return round((700*(np.exp(mel/1127)-1)).astype('float32'))

  def xywh2ttff(self, xywh):
    x, y, w, h = list(xywh)
    ts = round((x-w/2)*self.clip_length)
    te = round((x+w/2)*self.clip_length)
    fl = self.mel_to_freq(1-(y+h/2))
    fh = self.mel_to_freq(1-(y-h/2))
    return [ts, te, fl, fh]

  def detect(self, weights, step=1500, conf_thres=0.1, imgsz=480, targetfilepath=None, iou_thres=0.25, targetclasses=None):
    if self.model and self.model_path == weights:
        pass
    else:
        self.model_path = weights
        self.model = YOLO(self.model_path)
        self.names = self.model.names
        self.soundclasses = pd.read_csv(
            self.model_path.replace('best.pt', 'soundclass.csv'),
            encoding='utf8',
            index_col='soundclass_id'
        ).T.to_dict()
    #print(self.model.names)
    if targetfilepath and os.path.exists(targetfilepath):
        self.rainbow_img = np.array(Image.open(targetfilepath).convert("RGB"))[:, :, ::-1]
    else:
        self.tfr(targetfilepath=targetfilepath, spect_type='rainbow')

    def iter_clips():
        """Yield one spectrogram window at a time to keep memory bounded."""
        for ts in range(0, self.duration, step):
            clip_start = round(ts/self.duration*self.rainbow_img.shape[1])
            clip_end = clip_start+round(self.clip_length/self.duration*self.rainbow_img.shape[1])
            if clip_end > self.rainbow_img.shape[1]:
                missing_width = clip_end - self.rainbow_img.shape[1]
                silence = np.full((self.rainbow_img.shape[0], missing_width, 3), 255, dtype=np.uint8)
                img0 = np.concatenate((self.rainbow_img[:, clip_start:], silence), axis=1)
                yield os.path.join(self.audiopath, self.audiofilename), img0, ts
                break
            yield (
                os.path.join(self.audiopath, self.audiofilename),
                self.rainbow_img[:, clip_start:clip_end],
                ts,
            )

    labels = [['file', 'classid', 'species_name', 'sound_class', 'scientific_name',
               "time_begin", "time_end", "freq_low", "freq_high", "score",
               "average_power_density", "SNR"]]
    if targetclasses:
        id2name = self.model.names
        name2id = {v: k for k, v in id2name.items()}
        classes = [name2id[item] for item in targetclasses.split(',')]
    else:
        classes = None
    for path, im0, time_start in iter_clips():
        results = self.model.predict(source=im0, imgsz=imgsz, conf=conf_thres, iou=iou_thres, verbose=False, classes=classes, device=self.device)

        for r in results:
            boxes = r.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                x = (x1 + x2) / 2 / im0.shape[1]
                y = (y1 + y2) / 2 / im0.shape[0]
                w = (x2 - x1) / im0.shape[1]
                h = (y2 - y1) / im0.shape[0]

                ttff = self.xywh2ttff([x, y, w, h])
                ts, te, fl, fh = ttff
                classid = int(self.names[cls])
                species_name = self.soundclasses[classid]['species_name']
                sound_class = self.soundclasses[classid]['sound_class']
                scientific_name = self.soundclasses[classid]['scientific_name']

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        average_power_density_dbfs, snr_db = signal_power(
                            self.analysis_audio,
                            (time_start + ts) / 1000,
                            (time_start + te) / 1000,
                            fl,
                            fh,
                        )
                except (ValueError, ZeroDivisionError):
                    average_power_density_dbfs = 'error'
                    snr_db = 'error'

                labels.append([
                    path, classid, species_name, sound_class, scientific_name,
                    round(time_start+ts), round(time_start+te),
                    fl, fh, round(conf, 3),
                    average_power_density_dbfs, snr_db
                ])

    return labels

def signal_power(audio, start_time, end_time, low_freq, high_freq):
  sr = audio.frame_rate

  # 取得 bit depth
  bit_depth = audio.sample_width * 8

  # 將音訊轉換為 NumPy 陣列
  samples = np.array(audio.get_array_of_samples())

  # 設定特定時間範圍（以秒為單位）
  start_sample = int(start_time * sr)
  end_sample = int(end_time * sr)
  y_segment = samples[start_sample:end_sample]
  if y_segment.size < 2:
    return 'error', 'error'

  # 設定特定頻率範圍（例如 300-3000 Hz）
  nyquist = 0.5 * sr
  low = low_freq / nyquist
  high = high_freq / nyquist
  low = max(float(low), 1e-6)
  high = min(float(high), 1 - 1e-6)
  if low >= high:
    return 'error', 'error'

  # 設計帶通濾波器
  b, a = signal.butter(4, [low, high], btype='band')
  filtered_signal = signal.lfilter(b, a, y_segment)

  # 計算功率密度譜
  nperseg = min(1024, y_segment.size)
  f, Pxx = signal.welch(y_segment, sr, nperseg=nperseg)

  # 選擇特定頻率範圍內的功率密度
  freq_mask = (f >= low_freq) & (f <= high_freq)
  Pxx_in_band = Pxx[freq_mask]

  # 計算特定頻率範圍內的平均功率密度
  average_power_density = np.mean(Pxx_in_band)
  if not np.isfinite(average_power_density) or average_power_density <= 0:
    return 'error', 'error'

  # 根據 bit depth 計算最大可能振幅
  max_possible_amplitude = 2 ** (bit_depth - 1)

  # 將平均功率密度轉換為 dB FS
  average_power_density_dbfs = 10 * np.log10(average_power_density / (max_possible_amplitude ** 2))

  # 計算訊號能量
  signal_power = np.mean(filtered_signal**2)

  # 假設噪聲（這裡用信號減去濾波後的信號作為噪聲估計）
  noise = y_segment - filtered_signal
  noise_power = np.mean(noise**2)
  if not np.isfinite(signal_power) or not np.isfinite(noise_power) or signal_power <= 0 or noise_power <= 0:
    return 'error', 'error'

  # 計算 SNR（以分貝為單位）
  snr_db = 10 * np.log10(signal_power / noise_power)
  return round(average_power_density_dbfs, 1), round(snr_db, 1)
  
def get_iou(bb1, bb2):
  """
  © https://github.com/MartinThoma/algorithms/blob/master/CV/IoU/IoU.py
  Calculate the Intersection over Union (IoU) of two bounding boxes.
  Parameters
  ----------
  bb : dict
      Keys: {'x1', 'x2', 'y1', 'y2'}
      The (x1, y1) position is at the top left corner,
      the (x2, y2) position is at the bottom right corner

  Returns
  -------
  float
      in [0, 1]
  """
  assert bb1['x1'] < bb1['x2']
  assert bb1['y1'] < bb1['y2']
  assert bb2['x1'] < bb2['x2']
  assert bb2['y1'] < bb2['y2']

  # determine the coordinates of the intersection rectangle
  x_left = max(bb1['x1'], bb2['x1'])
  y_top = max(bb1['y1'], bb2['y1'])
  x_right = min(bb1['x2'], bb2['x2'])
  y_bottom = min(bb1['y2'], bb2['y2'])

  if x_right < x_left or y_bottom < y_top:
      return 0.0, 0.0, 0.0

  # The intersection of two axis-aligned bounding boxes is always an
  # axis-aligned bounding box
  intersection_area = (x_right - x_left) * (y_bottom - y_top)

  # compute the area of both AABBs
  bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
  bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
  i_ration_bb1 = intersection_area / bb1_area
  i_ration_bb2 = intersection_area / bb2_area
  assert iou >= 0.0
  assert iou <= 1.0
  return iou, i_ration_bb1, i_ration_bb2

def merge_boxes(bb1, bb2):
  x1 = bb1['x1']
  x2 = bb1['x2']
  y1 = bb1['y1']
  y2 = bb1['y2']
  if bb2['x1'] < bb1['x1']:
    x1 = bb2['x1']
  if bb2['x2'] > bb1['x2']:
    x2 = bb2['x2']
  if bb2['y1'] < bb1['y1']:
    y1 = bb2['y1']
  if bb2['y2'] > bb1['y2']:
    y2 = bb2['y2']
  return {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2}

def clean_multi_boxes(audiofile, labels, threshold_iou=0.1, threshold_iratio=0.25, audio=None):
  df = pd.DataFrame(labels[1:],columns=labels[0])
  for col in ['average_power_density', 'SNR']:
    if col in df.columns:
      df[col] = df[col].astype(object)
  df = df.sort_values('time_begin')
  df_results = pd.DataFrame()
  soundclasses = df['classid'].unique()
  if audio is None:
    audio = AudioSegment.from_file(audiofile)
  if audio.channels > 1:
      audio = audio.split_to_mono()[0]  # 轉換為單聲道
  for classid in soundclasses:
    df_class = df[df['classid']==classid].reset_index(drop=True)
    for i in range(0, df_class.shape[0]):
      check = True
      bb1 = {'x1':df_class.loc[i, 'time_begin'], 'x2':df_class.loc[i, 'time_end'], 'y1':df_class.loc[i, 'freq_low'], 'y2':df_class.loc[i, 'freq_high']}
      for j in range(i+1, df_class.shape[0]):
        bb2 = {'x1':df_class.loc[j, 'time_begin'], 'x2':df_class.loc[j, 'time_end'], 'y1':df_class.loc[j, 'freq_low'], 'y2':df_class.loc[j, 'freq_high']}
        if bb2['x1'] >= bb1['x2']:
          break
        iou, i_ration_bb1, i_ration_bb2 = get_iou(bb1, bb2)
        i_ration = i_ration_bb1 if i_ration_bb1 > i_ration_bb2 else i_ration_bb2
        if iou >= threshold_iou or i_ration > threshold_iratio:
          score = df_class.loc[i, 'score']
          if df_class.loc[j, 'score'] > score:
            score = df_class.loc[j, 'score']
          merge_box = merge_boxes(bb1, bb2)
          try:
            with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=RuntimeWarning)
              average_power_density, SNR = signal_power(audio, merge_box['x1']/1000, merge_box['x2']/1000, merge_box['y1'], merge_box['y2'])
          except (ValueError, ZeroDivisionError):
            average_power_density = 'error'
            SNR = 'error'
          df_class.loc[j, 'time_begin'] = merge_box['x1']
          df_class.loc[j, 'time_end'] = merge_box['x2']
          df_class.loc[j, 'freq_low'] = merge_box['y1']
          df_class.loc[j, 'freq_high'] = merge_box['y2']
          df_class.loc[j, 'score'] = score
          df_class.loc[j, 'average_power_density'] = average_power_density
          df_class.loc[j, 'SNR'] = SNR
          check = False
          break
      if check:
        if df_results.shape[0] > 0:
          df_results = pd.concat([df_results, df_class[df_class.index == i]],axis=0, ignore_index=True) 
        else:
          df_results = df_class[df_class.index == i]
  return df_results.sort_values('time_begin').reset_index(drop=True)

def draw_labels(silic, labels, outputpath=None):
  if outputpath and os.path.isdir(outputpath):
    targetpath = os.path.join(outputpath, '%s.png'%silic.audiofilename_without_ext)
  else:
    if not os.path.isdir(os.path.join(silic.audiopath, 'labels')):
      os.makedirs(os.path.join(silic.audiopath, 'labels'))
    targetpath = os.path.join(silic.audiopath, 'labels', '%s.png'%silic.audiofilename_without_ext)
  outputimage = silic.tfr()
  img_pil = Image.open(outputimage)
  width, height = img_pil.size
  fontpath = os.path.join(PROJECT_DIR, 'model', 'wt011.ttf')
  font = ImageFont.truetype(fontpath, 9)
  draw = ImageDraw.Draw(img_pil)
  for index, label in labels.iterrows():
    x1 = round(label['time_begin']/silic.duration*width)
    x2 = round(label['time_end']/silic.duration*width)
    y1 = round((1-label['freq_high']/(silic.sr/2))*height)
    y2 = round((1-label['freq_low']/(silic.sr/2))*height)
    tag = '%s%s(%.3f)' %(label['species_name'], label['sound_class'], label['score'])
    draw.text((x1, y1-12),  tag, font = font, fill = 'red')
    draw.rectangle(((x1, y1), (x2, y2)), outline='red')
  try:
    img_pil.save(targetpath)
  except OSError:
    targetpath = '%spng' %targetpath[:-3]
    img_pil.save(targetpath)
  #img_pil.show()
  print(targetpath, 'saved')
  return targetpath

# Worker-side globals to cache model per process
G_SILIC = None

def _init_worker(weights: str):
  """
  Initializer that runs once per process. Pre-loads YOLO model into a Silic instance
  so each task in the same process reuses the model (big speed-up).
  """
  global G_SILIC
  G_SILIC = Silic()
  # Preload model
  from ultralytics import YOLO as _YOLO
  G_SILIC.model_path = weights
  G_SILIC.model = _YOLO(G_SILIC.model_path)
  G_SILIC.names = G_SILIC.model.names
  G_SILIC.soundclasses = pd.read_csv(
      G_SILIC.model_path.replace('best.pt', 'soundclass.csv'),
      encoding='utf8',
      index_col='soundclass_id'
  ).T.to_dict()

def _write_raven_table(labels, target_path):
  raven = labels.copy()
  raven['Selection'] = range(1, len(raven) + 1)
  raven['View'] = 'Spectrogram 1'
  raven['Channel'] = '1'
  raven['Begin Time (s)'] = raven['time_begin'] / 1000
  raven['End Time (s)'] = raven['time_end'] / 1000
  raven['Low Freq (Hz)'] = raven['freq_low']
  raven['High Freq (Hz)'] = raven['freq_high']
  raven['Delta Time (s)'] = raven['End Time (s)'] - raven['Begin Time (s)']
  raven['Delta Freq (Hz)'] = raven['High Freq (Hz)'] - raven['Low Freq (Hz)']
  raven['Avg Power Density (dB FS/Hz)'] = raven['average_power_density']
  raven['Annotation'] = raven.apply(
      lambda row: f"{row['species_name']} ({row['scientific_name']}) : "
                  f"{row['sound_class']}, Score: {row['score']}",
      axis=1,
  )
  columns = [
      'Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)',
      'Low Freq (Hz)', 'High Freq (Hz)', 'Delta Time (s)', 'Delta Freq (Hz)',
      'Avg Power Density (dB FS/Hz)', 'Annotation',
  ]
  raven[columns].to_csv(target_path, index=False, sep='\t', encoding='big5', errors='ignore')

def _run_one_file(silic, audiofile, result_paths, conf_thres, step, targetclasses):
  """Process one recording; shared by serial and multiprocessing execution."""
  linear_path, rainbow_path, label_path, raven_path, audio_path = result_paths
  silic.audio(audiofile)

  if audio_path:
    target_audio = os.path.join(audio_path, silic.audiofilename)
    if os.path.abspath(audiofile) != os.path.abspath(target_audio):
      shutil.copy2(audiofile, target_audio)

  linear_png = os.path.join(linear_path, silic.audiofilename_without_ext + '.png')
  if not os.path.exists(linear_png):
    silic.tfr(targetfilepath=linear_png, spect_type='linear')

  rainbow_png = os.path.join(rainbow_path, silic.audiofilename_without_ext + '.png')
  labels = silic.detect(
      weights=silic.model_path,
      step=step,
      targetclasses=targetclasses,
      conf_thres=conf_thres,
      targetfilepath=rainbow_png,
  )
  if len(labels) == 1:
    return {
        'audiofile': audiofile, 'found': 0, 'species': 0, 'csv_path': None,
        'message': f'No sound found in {audiofile}.',
    }

  newlabels = clean_multi_boxes(audiofile, labels, audio=silic.analysis_audio)
  newlabels['file'] = silic.audiofilename
  csv_path = os.path.join(label_path, silic.audiofilename_without_ext + '.csv')
  newlabels.to_csv(csv_path, index=False, encoding='utf-8-sig')
  raven_path_txt = os.path.join(raven_path, silic.audiofilename_without_ext + '_selections.txt')
  _write_raven_table(newlabels, raven_path_txt)
  species = int(newlabels['classid'].nunique())
  return {
      'audiofile': audiofile,
      'found': int(newlabels.shape[0]),
      'species': species,
      'csv_path': csv_path,
      'message': f'{newlabels.shape[0]} sounds of {species} species is/are found in {audiofile}',
  }

def _process_one_file(args):
  """
  Run full pipeline for a single file inside a worker process.
  Returns a dict summarizing results and path to per-file CSV (if any).
  """
  (audiofile, result_paths, conf_thres, step, targetclasses) = args
  try:
    return _run_one_file(G_SILIC, audiofile, result_paths, conf_thres, step, targetclasses)

  except Exception as e:
    return {
      'audiofile': audiofile,
      'found': 0,
      'species': 0,
      'csv_path': None,
      'message': f"Error when processing {audiofile}: {str(e)}"
    }

def browser(source, model="", step=1500, targetclasses='', conf_thres=0.1, savepath='result_silic', zip=False, workers=1, ui_callback=None, progress_cb=None):
  """
  Main pipeline used by both CLI and GUI. When workers > 1, run in parallel.
  ui_callback: optional callable(str) to stream progress messages back to GUI.
  """
  def _log(msg):
    if ui_callback:
      try:
        ui_callback(msg)
        return
      except Exception:
        pass
    print(msg)

  def _progress(done, total):
    if progress_cb:
      try:
        progress_cb(int(done), int(total))
      except Exception:
        pass

  # Parse targetclasses
  if not targetclasses:
    targetclasses = ''
  elif isinstance(targetclasses, (list, tuple, np.ndarray)):
    targetclasses = ",".join(str(item) for item in targetclasses)
  else:
    targetclasses = str(targetclasses)

  t0 = time.time()

  # Prepare result directories
  result_path = savepath or 'result_silic'

  same_source_and_result = (
      os.path.isdir(source)
      and os.path.normcase(os.path.abspath(source))
      == os.path.normcase(os.path.abspath(result_path))
  )
  if same_source_and_result:
    audio_path = None
  else:
    audio_path = os.path.join(result_path, 'audio')

  linear_path = os.path.join(result_path, 'linear')
  rainbow_path = os.path.join(result_path, 'rainbow')
  lable_path = os.path.join(result_path, 'label')
  js_path = os.path.join(result_path, 'js')
  raven_path = os.path.join(result_path, 'raven')

  for pth in [result_path, linear_path, rainbow_path, lable_path, js_path, raven_path] + ([audio_path] if audio_path else []):
    if pth and not os.path.isdir(pth):
      os.makedirs(pth, exist_ok=True)

  # Copy browser assets if available
  try:
    shutil.copyfile(os.path.join(PROJECT_DIR, 'browser', 'index.html'), os.path.join(result_path, 'index.html'))
  except Exception:
    pass

  # Collect media files
  media_files = get_media_files(source)
  if len(media_files) == 0:
    _log('No media files found.')
    return
  _log(f"SILIC Detector: {len(media_files)} files are found.")
  _progress(0, len(media_files))

  weights = os.path.join(PROJECT_DIR, 'model', model, 'best.pt')
  if not os.path.isfile(weights):
    raise FileNotFoundError(f'Model weights not found: {weights}')
  result_paths = (linear_path, rainbow_path, lable_path, raven_path, audio_path)
  if os.path.isfile(source):
    abs_files = [os.path.abspath(source)]
  else:
    abs_files = [os.path.join(source, f) for f in media_files]

  csv_paths = []
  if workers and workers > 1:
    _log(f"Using {workers} processes...")
    with ProcessPoolExecutor(max_workers=int(workers), initializer=_init_worker, initargs=(weights,)) as ex:
      futures = {ex.submit(_process_one_file, (afile, result_paths, conf_thres, step, targetclasses)): afile for afile in abs_files}
      done = 0
      total = len(futures)
      for fut in as_completed(futures):
        res = fut.result()
        done += 1
        _progress(done, total)
        _log(res['message'])
        if res.get('csv_path'):
          csv_paths.append(res['csv_path'])
  else:
    # Serial path: reuse a single Silic instance and YOLO model in this process
    model_obj = Silic()
    model_obj.model_path = weights
    model_obj.model = YOLO(model_obj.model_path)
    model_obj.names = model_obj.model.names
    model_obj.soundclasses = pd.read_csv(
        model_obj.model_path.replace('best.pt', 'soundclass.csv'),
        encoding='utf8',
        index_col='soundclass_id'
    ).T.to_dict()

    total = len(abs_files)
    for done, audiofile in enumerate(abs_files, start=1):
      try:
        res = _run_one_file(model_obj, audiofile, result_paths, conf_thres, step, targetclasses)
      except Exception as exc:
        res = {'csv_path': None, 'message': f'Error when processing {audiofile}: {exc}'}
      _progress(done, total)
      _log(res['message'])
      if res.get('csv_path'):
        csv_paths.append(res['csv_path'])

  # Aggregate results across files
  if not csv_paths:
    _log('No sounds found!')
    return

  all_dfs = []
  for pth in csv_paths:
    try:
      all_dfs.append(pd.read_csv(pth))
    except Exception:
      pass
  if not all_dfs:
    _log('No sounds found!')
    return

  all_labels = pd.concat(all_dfs, ignore_index=True)
  all_labels.to_csv(os.path.join(lable_path, 'labels.csv'), index=False, encoding='utf-8-sig')

  i = len(csv_paths)
  _log(f"{all_labels.shape[0]} sounds of {len(all_labels['classid'].unique())} species is/are found in {i} recording(s). Preparing the browser package ...")

  df_classes = pd.read_csv(weights.replace('best.pt', 'soundclass.csv'))
  if len(targetclasses) > 0:
    targetclasses = [int(item) for item in targetclasses.split(',')]
    df_classes = df_classes[df_classes['soundclass_id'].isin(targetclasses)]
  else:
    names = all_labels['classid'].unique()
    df_classes = df_classes[df_classes['soundclass_id'].isin(names)]
  sounds = {
      str(row['soundclass_id']): [row['species_name'], row['sound_class'], row['scientific_name']]
      for _, row in df_classes.iterrows()
  }
  with open(os.path.join(js_path, 'soundclass.js'), 'w', newline='', encoding='utf-8') as csv_file:
    csv_file.write('var sounds = ')
    json.dump(sounds, csv_file, ensure_ascii=False)
    csv_file.write(';')

  with open(os.path.join(js_path, 'labels.js'), 'w', newline='', encoding='utf-8') as f:
    labels_data = all_labels[
        [
            'file', 'time_begin', 'time_end', 'freq_low', 'freq_high',
            'classid', 'score', 'average_power_density', 'SNR',
        ]
    ].values.tolist()
    f.write('var labels = ')
    json.dump(labels_data, f, ensure_ascii=False)
    f.write(';\n')

  _log(f'Finished. All results were saved in the folder {result_path}')
  if zip:
    archive_path = shutil.make_archive(os.path.abspath(result_path), 'zip', root_dir=result_path)
    _log(f'Archive saved to {archive_path}')
  _log(f'{str(time.time()-t0)} used.')
  try:
    os.startfile(result_path)
  except Exception:
    pass

def parse_opt():
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=str, help='Source of a single file or 1-level folder')
  parser.add_argument('--model', type=str, default="", help='Version of model wights')
  parser.add_argument('--step', type=int, default=1500, help='Length of sliding window in ms.')
  parser.add_argument('--workers', type=int, default=1, help='')
  parser.add_argument('--targetclasses', type=str, default='', help='filter by class, comma-separated')
  parser.add_argument('--conf_thres', type=float, default=0.1, help='Threshold of confidence score from 0.0 to 1.0')
  parser.add_argument('--savepath', type=str, default='result_silic', help='Target folder of inference results archived')
  parser.add_argument('--zip', action='store_true', help='ZIP')
  opt = parser.parse_args()
  return opt

if __name__ == '__main__':
  opt = parse_opt()
  browser(**vars(opt))
