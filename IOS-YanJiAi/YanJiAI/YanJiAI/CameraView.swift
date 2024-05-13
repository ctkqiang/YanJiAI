//
//  CameraView.swift
//  IOSYanjiAi
//
//  Created by Cheng Tze Keong on 2024/5/12.
//

#if canImport(CoreML)
import CoreML
#endif

#if canImport(SwiftUI)
import SwiftUI
#endif

#if canImport(UIKit)
import UIKit
#endif

#if canImport(AVFoundation)
import AVFoundation
#endif

#if canImport(Vision)
import Vision
#endif

#if canImport(CoreMedia)
import CoreMedia
#endif

#if canImport(Photos)
import Photos
#endif

class CameraView: UIViewController{
    public var isDoubleTapped = false
    
    private var sentimentModel: SentimentModel!
    
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    
    private var drawings: [CAShapeLayer] = []
    private var isSentimentAnalysisAvailable: Bool = false
    
    private let movieOutput = AVCaptureMovieFileOutput()
    private var isRecording: Bool = false
    
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
    
    public override func viewDidLoad() {
        super.viewDidLoad()
        
        self.loadSentimentModel()
        
        self.addCameraInput()
        self.showCameraInput()
        
        self.getCameraFrame()
        
        self.captureSession.startRunning()
    }
    
    public override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        self.previewLayer.frame = view.frame
    }
    
    private func reloadCameraInput() -> Void {
          
        if let currentInput = self.captureSession.inputs.first {
            self.captureSession.removeInput(currentInput)
        }
            
        self.addCameraInput()
    }
    
    
    public func toggleCamera() -> Void {
        self.isDoubleTapped.toggle()
        self.reloadCameraInput()
    }
    
    private func analyzeSentiment(observedFaces: [VNFaceObservation]) -> Void{
        guard let prediction = try? sentimentModel.model.prediction(from: observedFaces as! MLFeatureProvider) else {
            NSLog("情感预测失败")
            return
        }
            
        let sentiment = prediction
            
        NSLog("情感: \(sentiment)")
            
        // Process sentiment and display the result
    }
    
    private func loadSentimentModel() {
        guard let modelURL = Bundle.main.url(forResource: "SentimentModel", withExtension: "mlmodelc") else {
            fatalError("无法加载 SentimentModel.mlmodelc")
        }
            
        do {
            sentimentModel = try SentimentModel(contentsOf: modelURL)
        } catch {
            fatalError("无法加载模型: \(error.localizedDescription)")
        }
    }
    
    private func addCameraInput() -> Void {
        guard let device = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInTrueDepthCamera, .builtInDualCamera, .builtInWideAngleCamera],
            mediaType: .video,
            position: (self.isDoubleTapped) ? .back : .front
        ).devices.first else {
            fatalError("摄像头启动失败了。")
        }
        
        let cameraInput = try! AVCaptureDeviceInput(device: device)
        
        self.captureSession.addInput(cameraInput)
    }
    
    private func showCameraInput() -> Void {
        self.previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
    }
    
    private func getCameraFrame() -> Void {
        self.videoDataOutput.videoSettings = [
            (kCVPixelBufferPixelFormatTypeKey as NSString) : NSNumber(value:kCVPixelFormatType_32BGRA)
        ] as [String: Any]
        
        self.videoDataOutput.alwaysDiscardsLateVideoFrames = true
        self.videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label:"camera_frame_processing_queue"))
        
        self.captureSession.addOutput(videoDataOutput)
        
        guard let connection = videoDataOutput.connection(with: .video), connection.isVideoOrientationSupported else {
            return
        }
        
        connection.videoOrientation = .portrait
    }
    
    private func startRecording() -> Void {
        guard !isRecording else { return }
        
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent(
            "yanjiai_\(self.randomString(length: 10)).mov"
        )
        
//        self.movieOutput.startRecording(to: outputURL, recordingDelegate: self)
        self.isRecording = true
    }
    
    private func stopRecording() -> Void {
        guard isRecording else { return }
            
        self.movieOutput.stopRecording()
        self.isRecording = false
    }
    
    private func randomString(length: Int) -> String {
        let letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        return String((0..<length).map { _ in letters.randomElement()! })
    }

    private func detectFaces(image: CVPixelBuffer) -> Void {
        let faceDetectedRequest = VNDetectFaceLandmarksRequest(completionHandler: {vnRequest, err in
            DispatchQueue.main.async {
                if let results = vnRequest.results as? [VNFaceObservation], results.count > 0 {
                    self.handleFaceDetectionResults(observedFaces: results)
                    
                    if self.isSentimentAnalysisAvailable {
                        self.analyzeSentiment(observedFaces: results)
                    }
                    
                    NSLog("✅ 检测到 \(results.count) 张人脸")
                    self.showToast(message: "✅ 检测到 \(results.count) 张人脸")
                } else {
                    NSLog("❌ 未检测到人脸")
                    self.showToast(message: "❌ 未检测到人脸")
                }
            }
        })
        
        let imageResultHandler = VNImageRequestHandler(cvPixelBuffer: image, orientation: .leftMirrored, options: [:])
        
        try? imageResultHandler.perform([faceDetectedRequest])
        
    }
    
    private func handleFaceDetectionResults(observedFaces: [VNFaceObservation]) -> Void {
        self.clearDrawings()
        
        let faceBoundingsBoxes: [CAShapeLayer] = observedFaces.map({ (observedFace: VNFaceObservation) ->
            CAShapeLayer in
            
            let faceBoundBoxOnScreen = self.previewLayer.layerRectConverted(fromMetadataOutputRect: observedFace.boundingBox)
            let faceBoundBoxPath = CGPath(rect: faceBoundBoxOnScreen, transform: nil)
            let faceBoundShape = CAShapeLayer()
            let faceBoundShape2 = CAShapeLayer()
            let textLayer = CATextLayer()
            
            textLayer.string = "✅ 人类"
            textLayer.foregroundColor = UIColor.white.cgColor
            textLayer.fontSize = 20
            textLayer.isHidden = false
            textLayer.alignmentMode = .left
            textLayer.frame = CGRect(
                x: faceBoundBoxOnScreen.origin.x,
                y: faceBoundBoxOnScreen.origin.y - 30,
                width: faceBoundBoxOnScreen.size.width,
                height: 50
            )
            
            faceBoundShape.path = faceBoundBoxPath
            faceBoundShape.fillColor = UIColor.clear.cgColor
            faceBoundShape.strokeColor = UIColor.systemPink.cgColor
            faceBoundShape.isHidden = false
            faceBoundShape.lineWidth = 3
            faceBoundShape.bounds = CGRect.zero
            faceBoundShape.allowsEdgeAntialiasing = true
            faceBoundShape.addSublayer(textLayer)
            
            return faceBoundShape
        })
        
        
        faceBoundingsBoxes.forEach { faceBoundingBox in
            view.layer.addSublayer(faceBoundingBox)
            
            self.drawings = faceBoundingsBoxes
        }
    }
        
    private func clearDrawings() -> Void {
        self.drawings.forEach({ drawing in
            drawing.removeFromSuperlayer()
        })
    }
    
    public func showToast(message: String) {
        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let window = windowScene.windows.first else { return }
        
        let label = UILabel()
        
        label.textAlignment = .center
        label.text = message
        label.textColor = .white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        label.numberOfLines = 0
        label.layer.cornerRadius = 10
        label.clipsToBounds = true
        
        let padding: CGFloat = 10
        let maxSize = CGSize(
            width: window.frame.size.width * 0.8 - 2 * padding,
            height: window.frame.size.height * 0.8 - 2 * padding
        )
        let expectedSize = label.sizeThatFits(maxSize)
        
        label.frame = CGRect(
            x: (window.frame.size.width - expectedSize.width) / 2,
            y: window.frame.size.height - expectedSize.height - 50 - padding,
            width: expectedSize.width + 2 * padding,
            height: expectedSize.height + 2 * padding
        )
        
        window.addSubview(label)
        
        UIView.animate(withDuration: 2.0, delay: 1.0, options: .curveEaseOut, animations: {
            label.alpha = 0.0
        }, completion: { _ in label.removeFromSuperview() })
    }
}

struct CameraViewWrapper: UIViewControllerRepresentable {
    @State private var isDoubleTapped: Bool = false
    
    public func makeUIViewController(context: Context) -> CameraView {
        let cameraView = CameraView()
                
        // Add gesture recognizer for double tap
        let tapGesture = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleDoubleTap)
        )
                
        tapGesture.numberOfTapsRequired = 2
        cameraView.view.addGestureRecognizer(tapGesture)
                
        return cameraView
    }
    
    public func makeCoordinator() -> Coordinator {
        Coordinator(isDoubleTapped: $isDoubleTapped)
    }
    
    public func updateUIViewController(_ uiViewController: CameraView, context: Context) {
        uiViewController.isDoubleTapped = isDoubleTapped
        
    }
    
    class Coordinator: NSObject {
        @Binding var isDoubleTapped: Bool
            
        init(isDoubleTapped: Binding<Bool>) {
            _isDoubleTapped = isDoubleTapped
        }
            
        @objc func handleDoubleTap() {
            isDoubleTapped.toggle()
        }
    }
}


extension CameraView: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        guard let frame = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            NSLog("样本缓冲为空")
            return
        }
        
        self.detectFaces(image: frame)
    }
}
