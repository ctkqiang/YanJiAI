//
//  ContentView.swift
//  YanJiAI
//
//  Created by Cheng Tze Keong on 2024/5/13.
//

import SwiftUI

struct ContentView: View {
    @State private var isDoubleTapped: Bool = false
    
    var body: some View {
        let cameraView: CameraView = CameraView(isDoubleTapped: $isDoubleTapped)
        
        VStack {
            CameraViewWrapper(isDoubleTapped: self.$isDoubleTapped).onAppear {
                NSLog("🤖 眼迹AI准备就绪 🤖")
                cameraView.showToast(message:"🤖 眼迹AI准备就绪 🤖")
            }
            .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
            .edgesIgnoringSafeArea(.all)
            .onTapGesture {
                NSLog("✅ 在点击...")
                
                cameraView.toggleCamera()
                cameraView.isDoubleTapped.toggle()
            }
            .onTapGesture(count: 2, perform: {
                cameraView.toggleCamera()
            })
        }
        .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
        .edgesIgnoringSafeArea(.all)
    }
}

#Preview {
    ContentView()
}
