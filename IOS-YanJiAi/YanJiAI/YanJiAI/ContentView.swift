//
//  ContentView.swift
//  YanJiAI
//
//  Created by Cheng Tze Keong on 2024/5/13.
//

import SwiftUI

struct ContentView: View {
    private var cameraView: CameraView = CameraView()
    
    var body: some View {
        VStack {
            CameraViewWrapper().onAppear {
                NSLog("✅")
                
                self.cameraView.showToast(message:"✅")
            }
            .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
            .edgesIgnoringSafeArea(.all)
        }
        .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
        .edgesIgnoringSafeArea(.all)
    }
}

#Preview {
    ContentView()
}
