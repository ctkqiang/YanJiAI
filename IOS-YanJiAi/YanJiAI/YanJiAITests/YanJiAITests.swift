//
//  YanJiAITests.swift
//  YanJiAITests
//
//  Created by Cheng Tze Keong on 2024/5/13.
//
import XCTest
@testable import YanJiAI

final class YanJiAITests: XCTestCase 
{
    
    var cameraView: CameraView!

    override func setUpWithError() throws 
    {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws 
    {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }
    
    override func setUp() 
    {
        super.setUp()
        // Initialize CameraView instance
        cameraView = CameraView(isDoubleTapped: .constant(false))
        cameraView.loadViewIfNeeded()
    }
    
    override func tearDown() 
    {
        // Clean up resources
        cameraView = nil
        super.tearDown()
    }
    
    func testCameraViewInit() 
    {
        // Test if CameraView is initialized correctly
        XCTAssertNotNil(cameraView)
    }
    
    func testCameraViewDidLoad() 
    {
        // Test if viewDidLoad() sets up the capture session
        cameraView.viewDidLoad()
        XCTAssertNotNil(cameraView.captureSession)
        XCTAssertTrue(cameraView.captureSession.isRunning)
    }
    
    func testToggleCamera() 
    {
        // Test if toggleCamera() toggles isDoubleTapped correctly
        XCTAssertFalse(cameraView.isDoubleTapped)
        cameraView.toggleCamera()
        XCTAssertTrue(cameraView.isDoubleTapped)
    }

    func testExample() throws 
    {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        // Any test you write for XCTest can be annotated as throws and async.
        // Mark your test throws to produce an unexpected failure when your test encounters an uncaught error.
        // Mark your test async to allow awaiting for asynchronous code to complete. Check the results with assertions afterwards.
    }

    func testPerformanceExample() throws 
    {
        // This is an example of a performance test case.
        self.measure 
        {
            // Put the code you want to measure the time of here.
        }
    }

}
