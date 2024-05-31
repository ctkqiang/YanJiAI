//
//  ObjectDetections.swift
//  YanJiAI
//
//  Created by Cheng Tze Keong on 2024/5/31.
//

#if canImport(Foundation)
import Foundation
#endif

#if canImport(UIKit)
import UIKit
#endif

#if canImport(Vision)
import Vision
#endif

public struct ObjectDetections
{
    private var model: VNCoreMLModel!
    private var request: VNCoreMLRequest!
    
    public init()
    {
        guard let modelURL = Bundle.main.url(forResource: "YOLOv3Int8LUT", withExtension: "mlmodelc") else
        {
            fatalError("Failed to load the model")
        }
        
        do 
        {
            let coreMLModel = try MLModel(contentsOf: modelURL)
            self.model = try VNCoreMLModel(for: coreMLModel)
            self.request = VNCoreMLRequest(model: self.model, completionHandler: self.handleDetection)
        }
        catch
        {
            fatalError("Failed to create VNCoreMLModel: \(error)")
        }
    }
    
    private func detectObjects(in image: UIImage) -> Void
    {
        guard let ciImage = CIImage(image: image) else 
        {
            return
        }

        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        
        do
        {
            try handler.perform([self.request])
        } 
        catch
        {
            NSLog("Failed to perform request: \(error)")
        }
    }
    
    private func handleDetection(request: VNRequest, error: Error?) -> Void
    {
        guard let results = request.results as? [VNRecognizedObjectObservation] else 
        {
            return
        }
        
        for result in results
        {
            let label = result.labels.first?.identifier ?? "Unknown"
            let confidence = result.labels.first?.confidence ?? 0.0
            let boundingBox = result.boundingBox
            
            NSLog("Detected object: \(label), Confidence: \(confidence), BoundingBox: \(boundingBox)")
            
            // You can now use this information to draw bounding boxes or display labels
        }
    }
}
