/// Copyright 2012-2025 (C) Butterfly Network, Inc.

import ButterflyImagingKit
import UIKit

@MainActor
class Model: ObservableObject {
    enum Stage {
        case startingUp, ready, updateNeeded, startingImaging, imaging
    }
    @Published var availablePresets: [ImagingPreset] = []
    @Published var colorGain = 0
    @Published var depth = Measurement.centimeters(0)
    @Published var depthBounds = Measurement.centimeters(0)...Measurement.centimeters(0)
    @Published var gain = 0
    @Published var image: UIImage?
    @Published var licenseState: ButterflyImaging.LicenseState = .invalid
    @Published var mode = UltrasoundMode.bMode
    @Published var preset: ImagingPreset?
    @Published var probe: Probe?
    @Published var stage = Stage.startingUp
    @Published var inProgress = false
    @Published var updating = false
    @Published var updateProgress: TimedProgress?
    @Published var buttonPressCounter = -2
    @Published var selectedServerName = "Will"
    @Published var serverIP = "192.168.1.83"

    @Published var alertError: Error? { didSet { showingAlert = (alertError != nil) } }
    @Published var showingAlert: Bool = false

    private let imaging = ButterflyImaging.shared

    static var shared: Model = Model()
    private init() {
        imaging.licenseStates = { [weak self] in
            self?.licenseState = $0
        }



        // Receive logs from the Butterfly Imaging SDK:
        imaging.clientLoggingCallback = { string, level in
            print("Butterfly SDK (level='\(level)': \(string)")
        }
        imaging.isClientLoggingEnabled = true
    }

    private var lastButtonPressCount = 0
    private var lastButtonPressCountTop = 0
    private var lastButtonPressCountBottom = 0

    func sendImageToServer(image: UIImage, depth: Measurement<UnitLength>) async {
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            showAlert(message: "Failed to convert image to JPEG")
            return
        }

        let url = URL(string: "http://\(serverIP):3000/capture")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("image/jpeg", forHTTPHeaderField: "Content-Type")
        
        // Create metadata as JSON
        let metadata = ["max_probe_depth": depth.value]
        let jsonData = try? JSONSerialization.data(withJSONObject: metadata)
        
        request.setValue("application/json", forHTTPHeaderField: "X-Metadata-Content-Type")
        request.setValue(String(data: jsonData ?? Data(), encoding: .utf8), forHTTPHeaderField: "X-Metadata")
        request.httpBody = imageData

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode) else {
                showAlert(message: "Server returned an error")
                return
            }
        } catch {
            showAlert(message: "Failed to send image: \(error.localizedDescription)")
        }
    }

    func sendSetExternalRequest() async {
        let url = URL(string: "http://\(serverIP):3003/setexternal")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        print("🔄 Attempting to send setexternal request to: \(url)")
        
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode) else {
                print("❌ Server returned an error for setexternal request. Status: \((response as? HTTPURLResponse)?.statusCode ?? -1)")
                return
            }
            print("✅ External reference pose set successfully")
        } catch {
            print("❌ Failed to send setexternal request: \(error.localizedDescription)")
        }
    }

    func setState(_ state: ImagingState, imagingStateChanges: ImagingStateChanges) {
        // Check for button press
        if state.probe.buttonPressedCountMain > lastButtonPressCount {
            lastButtonPressCount = state.probe.buttonPressedCountMain
            buttonPressCounter += 1
            
            // Get current image based on mode
            if let image = state.bModeImage?.image {
                Task {
                    await sendImageToServer(image: image, depth: state.depth)
                }
            } else if let image = state.mModeImage?.image {
                Task {
                    await sendImageToServer(image: image, depth: state.depth)
                }
            } else {
                showAlert(message: "No image available to send")
            }
        }
        
        // Debug: Print all available properties of the probe object (remove this once we find the right properties)
        let mirror = Mirror(reflecting: state.probe)
        for child in mirror.children {
            if let label = child.label, label.contains("button") || label.contains("Button") {
                print("Probe property: \(label) = \(child.value)")
            }
        }
        
        // Try to detect top and bottom button presses using reflection
        let mirror2 = Mirror(reflecting: state.probe)
        for child in mirror2.children {
            if let label = child.label {
                // Check for up button
                if label == "buttonPressedCountUp" {
                    if let currentCount = child.value as? Int, currentCount > lastButtonPressCountTop {
                        lastButtonPressCountTop = currentCount
                        print("🔼 Up button pressed (property: \(label)) - sending setexternal request")
                        Task {
                            await sendSetExternalRequest()
                        }
                    }
                }
                
                // Check for down button
                if label == "buttonPressedCountDown" {
                    if let currentCount = child.value as? Int, currentCount > lastButtonPressCountBottom {
                        lastButtonPressCountBottom = currentCount
                        print("🔽 Down button pressed (property: \(label)) - sending setexternal request")
                        Task {
                            await sendSetExternalRequest()
                        }
                    }
                }
            }
        }
        
        availablePresets = state.availablePresets
        colorGain = state.colorGain
        depth = state.depth
        depthBounds = state.depthBounds
        gain = state.gain
        mode = state.mode
        preset = state.preset
        probe = state.probe

        switch mode {
        case .bMode, .colorDoppler:
            if imagingStateChanges.bModeImageChanged,
               let img = state.bModeImage?.image {
                image = img
            }
        case .mMode:
            if imagingStateChanges.mModeImageChanged,
               let img = state.mModeImage?.image {
                image = img
            }
        @unknown default:
            break
        }

        switch stage {
        case .startingUp:
            stage = .ready
        case .ready:
            break
        case .updateNeeded:
            if state.probe.state != .firmwareIncompatible {
                stage = .ready
            }
        case .startingImaging:
            // If we have an image then we are "imaging".
            if image != nil {
                stage = .imaging
            }
            // If the user disconnects the probe while we are starting imaging, revert.
            if state.probe.state == .disconnected {
                stopImaging()
            }
        case .imaging:
            if state.probe.state == .disconnected {
                stopImaging()
            }
        }

        if state.probe.state == .firmwareIncompatible {
            stage = .updateNeeded
        }
    }

    func startImaging(preset: ImagingPreset? = nil, depth: Double? = nil) {
        stage = .startingImaging
        var parameters: PresetParameters? = nil

        // Send custom initial depth if it represents a change from the default one.
        if
            let preset,
            let depth,
            preset.defaultDepth.converted(to: .centimeters).value != depth
        {
            parameters = PresetParameters(depth: .centimeters(depth))
        }
        Task {
            do {
                try await imaging.startImaging(preset: preset, parameters: parameters)
            } catch {
                alertError = error
                print("Failed to startImaging, with error: \(error)")
            }
        }
    }

    func connectSimulatedProbe() async {
        inProgress = true
        await imaging.connectSimulatedProbe()
        inProgress = false
    }

    func disconnectSimulatedProbe() async {
        inProgress = true
        await imaging.disconnectSimulatedProbe()
        inProgress = false
    }

    func startup(clientKey: String) async throws {
        do {
            try await imaging.startup(clientKey: clientKey)

            // (Optional) setting up local client logging.
            imaging.isClientLoggingEnabled = true
            imaging.clientLoggingCallback = { message, severity in
                 print("[ButterflyImagingKitExample]: [\(severity)] \(message)")
            }
        } catch {
            alertError = error
            print("Failed to start up backend, with error: \(error)")
        }
    }

    func stopImaging() {
        stage = .ready
        imaging.stopImaging()
        image = nil
    }

    /// Starts the firmware update process.
    ///
    /// Note to ask user to keep the probe plugged-in during the update.
    func updateFirmware() async {
        updating = true
        do {
            for try await progress in imaging.updateFirmware() {
                updateProgress = progress
            }
            print("Update finished.")
        } catch {
            print("Update error: \(error)")
        }
        updating = false
    }

    func clearError() {
        alertError = nil
    }

    func showAlert(message: String) {
        alertError = NSError(domain: "", code: 0, userInfo: [NSLocalizedDescriptionKey: message])
    }

    func resetButtonPressCounter() {
        buttonPressCounter = -2
    }
    
    func incrementButtonPressCounter() {
        buttonPressCounter += 1
    }
    
    func getAvailableServers() -> [String: String] {
        return [
            "Monty 1": "192.168.1.83"
        ]
    }
    
    func getServerNames() -> [String] {
        return ["Monty 1"]
    }
    
    func setServerFromName(_ name: String) {
        let servers = getAvailableServers()
        if let ip = servers[name] {
            selectedServerName = name
            serverIP = ip
        }
    }
}
