# Vive Tracker Server

Here we describe how to setup a Windows computer with a Vive Tracker 3.0 device.

This will make use of SteamVR and Valve Base Stations to monitor where the Tracker is, and
then stream this via a server to the Mac that is running a Monty experiment.

## Requirements

- Windows 10+ PC with the latest version of Steam and SteamVR installed
- `openvr`
- One (ideally three) SteamVR Base Stations 2.0 ("Lighthouses")
- A Vive Tracker 3.0, and its associated USB dongle.

## Steps

### Background Setup
- On the Windows PC, install the latest version of Steam
- In the Steam app, install SteamVR (~5 GB)
    - After installation, launch SteamVR at least once to ensure it’s set up (you’ll see the VR status window, which will likely complain about no headset – we will address that soon).
- Setup a Python environment on your Windows computer, e.g. with Miniconda
    - We use an older version of Miniconda with Python 3.8 support to enable compatibility with Monty v0.5.0
    - An installer can be found here [Miniconda3-py38_23.11.0-2-Windows-x86_64.exe](https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Windows-x86_64.exe)
    - Note `conda` on Windows expects you to use the Anaconda Powershell Prompt, rather than the standard Windows Powershell to interact with it, so you need to launch this when activating your environment
- Run `pip install openvr`

### Setup Base Stations
- Plug in the base stations to power them on, and ensure they can see where the HTC Vive Tracker will be positioned
    - Note these serve as waypoints for the Tracker to detect, but they do not need WiFi or Bluetooth to connect to the PC or the Tracker itself - they are simply used as prominent signal sources for the Tracker, and so they only need a power source


### Configuring SteamVR for Headset-Free Operation
By default, SteamVR expects a VR headset to be connected. We need to tweak some settings to run SteamVR in “headless” or no-HMD mode so that it will monitor just our tracker. Most of the below instructions are based on [the post here](https://www.notion.so/yeove/Using-SteamVR-without-a-VR-headset-f7ed4268708a42c787d1628768e61d35).

#### 1. Enable the Null Driver (virtual headset)

SteamVR includes a “null” driver which emulates a head-mounted device. Enabling this can help SteamVR run smoothly without a physical HMD.
- Open: <Steam install path>\steamapps\common\SteamVR\drivers\null\resources\settings\default.vrsettings (By default, Steam is in C:\Program Files (x86)\Steam\) 
- Find the "enable" key and set it to true


#### 2. Edit Core SteamVR Settings

Next, open your SteamVR user config file: <Steam install path>\resources\settings\default.vrsettings. Under the "steamvr" section, add or modify the following keys:

```
"forcedDriver": "null",
"activateMultipleDrivers": true,
"requireHmd": false
```

### Pair the Tracker with the Base Stations

#### Power on the Tracker
- Press the tracker’s Power button. A short press (1 second) turns it on. The LED should start blue then turn green if it has been paired before. If it hasn’t been paired to the dongle/PC yet, it will blink blue seeking connection.

#### Plug the Vive Tracker USB dongle into the PC
- Initiate Pairing Mode: If the tracker isn’t automatically connecting, put it into pairing mode. Hold down the tracker's power button for about 2 seconds until the LED starts blinking blue (this indicates it's in pairing mode). In the SteamVR status window on your PC, click the menu and navigate to Devices > Pair Controller. (Trackers are paired through the same interface used for controllers.)

#### Dongle Association:
- The tracker is now linked to that specific USB dongle. You typically only need to pair once. In the future, turning the tracker on (while SteamVR is running and the dongle is plugged in) should automatically connect it, even on a different PC.
- If it is working, you should see an icon for the Tracker puck and both base stations appear in the SteamVR window in blue.

#### Trouble-Shooting
- Base Station Channels: Make sure each Lighthouse is on a unique channel to avoid interference. SteamVR may auto-configure this, but you can verify via SteamVR > Devices > Base Station Settings. If needed, manually set one base station to channel 1 and the other to channel 2 (or use “Automatic Configuration” in SteamVR). Base Stations packaged as a set are usually pre-set to channels 1 and 2.
- Room Setup: You can safely ignore messages about running Room Setup for the “headset” if prompted. If you do end up running room setup, select the option for “Standing Only”.

### Data Streaming & Guidance Visualization

- We will be using this library, which provides Python bindings for Valve’s openVR library: https://github.com/cmbruns/pyopenvr
- If you haven't already, create a new conda environment on your Windows computer, activate it, and run `pip install openvr`

To stream the probe positions to a HTTP endpoint that Monty on a Mac can access
- Update your environment variable `VIVE_SERVER_URL` on the Mac computer to match the IPv4 address of the Windows PC
- On the Windows laptop with OpenVR & SteamVR running, dongle inserted etc, run `htc_vive/server.py`
- This provides a HTTP endpoint that provides the pose of the Vive Tracker.

#### Trouble-Shooting
- Ensure the Windows PC and Macbook are connected to the same WiFi, in order for the server to work as expected