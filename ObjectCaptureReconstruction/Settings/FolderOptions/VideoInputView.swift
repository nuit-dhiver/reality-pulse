/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose a video file and extract its frames for use as photogrammetry input.
Extracted JPEG frames are written to a temporary directory which is then set
as the image folder on the job draft.
*/

import AVFoundation
import SwiftUI
import UniformTypeIdentifiers
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "VideoInputView")

struct VideoInputView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    @State private var showFileImporter = false
    @State private var thumbnail: CGImage?
    @State private var extractionProgress: Double = 0
    @State private var isExtracting = false
    @State private var extractedFrameCount: Int?
    @State private var extractionTask: Task<Void, Never>?
    /// Tracks whether the current `draft.videoFile` URL has an active security scope.
    @State private var isAccessingSecurityScope = false

    var body: some View {
        LabeledContent("Video File:") {
            VStack(spacing: 6) {
                HStack {
                    Text(statusTitle)
                        .foregroundStyle(.secondary)
                        .font(.caption)

                    Spacer()

                    if draft.videoFile != nil {
                        Button {
                            clearVideo()
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .frame(height: 15)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                    }
                }
                .padding([.leading, .trailing], 6)
                .padding(.top, 3)
                .frame(height: 20)

                Divider()
                    .padding(.top, -4)
                    .padding(.horizontal, 6)

                HStack {
                    if isExtracting {
                        ProgressView(value: extractionProgress)
                            .progressViewStyle(.linear)
                            .padding(.horizontal, 8)
                    } else if let thumbnail {
                        Image(decorative: thumbnail, scale: 1.0)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 45, height: 45)
                            .clipped()
                            .cornerRadius(6)
                    } else {
                        Image(systemName: "video")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 28)
                            .foregroundStyle(.tertiary)
                    }
                }
                .frame(height: 35)

                Button {
                    logger.log("Opening an interface for selecting the video file...")
                    showFileImporter.toggle()
                } label: {
                    HStack {
                        if let videoFile = draft.videoFile {
                            Image(nsImage: NSWorkspace.shared.icon(forFile: videoFile.path))
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                            Text(videoFile.lastPathComponent)
                        } else {
                            Text("Choose Video...")
                        }
                        Spacer()
                    }
                }
                .padding(6)
                .disabled(isExtracting)
                .fileImporter(
                    isPresented: $showFileImporter,
                    allowedContentTypes: [.movie, .video, .mpeg4Movie, .quickTimeMovie]
                ) { result in
                    switch result {
                    case .success(let url):
                        let gotAccess = url.startAccessingSecurityScopedResource()
                        selectVideo(url, releaseScopeOnClear: gotAccess)
                    case .failure(let error):
                        draft.alertMessage = "\(error)"
                        draft.hasError = true
                    }
                }
            }
            .background(Color.gray.opacity(0.1))
            .cornerRadius(10)
            .onAppear {
                // Restore the thumbnail when the view reappears after extraction has already
                // completed (e.g. the user scrolled away and back). Skip if extraction is
                // still running since `selectVideo` will set the thumbnail on completion.
                guard !isExtracting, thumbnail == nil, let videoFile = draft.videoFile else { return }
                Task { thumbnail = await VideoHelper.thumbnail(for: videoFile) }
            }
        }
        .frame(height: 130)
        .dropDestination(for: URL.self) { items, _ in
            guard let url = items.first, VideoHelper.isVideoFile(url) else {
                logger.info("Dragged item is not a supported video file.")
                return false
            }
            selectVideo(url, releaseScopeOnClear: false)
            return true
        }
        .onDisappear {
            extractionTask?.cancel()
            releaseSecurityScope()
        }
    }

    // MARK: - Helpers

    private var statusTitle: String {
        if isExtracting {
            return "Extracting frames… \(Int(extractionProgress * 100))%"
        } else if let count = extractedFrameCount {
            return "\(count) Frames Extracted"
        } else {
            return "Drag in a video file"
        }
    }

    private func selectVideo(_ url: URL, releaseScopeOnClear: Bool) {
        extractionTask?.cancel()
        releaseSecurityScope()

        draft.videoFile = url
        draft.imageFolder = nil
        draft.boundingBoxAvailable = false
        extractedFrameCount = nil
        isExtracting = true
        extractionProgress = 0
        thumbnail = nil
        isAccessingSecurityScope = releaseScopeOnClear

        extractionTask = Task {
            do {
                let (outputDir, count) = try await VideoHelper.extractFrames(
                    from: url,
                    framesPerSecond: 2.0
                ) { @MainActor progress in
                    self.extractionProgress = progress
                }

                if Task.isCancelled { return }

                if count == 0 {
                    draft.alertMessage = "No frames could be extracted from the selected video."
                    draft.hasError = true
                    draft.videoFile = nil
                    draft.imageFolder = nil
                } else {
                    draft.imageFolder = outputDir
                    extractedFrameCount = count
                    thumbnail = await VideoHelper.thumbnail(for: url)
                }
            } catch {
                if Task.isCancelled { return }
                logger.warning("Frame extraction failed: \(error)")
                draft.alertMessage = error.localizedDescription
                draft.hasError = true
                draft.videoFile = nil
                draft.imageFolder = nil
            }

            isExtracting = false
        }
    }

    private func clearVideo() {
        extractionTask?.cancel()
        releaseSecurityScope()
        draft.videoFile = nil
        draft.imageFolder = nil
        draft.boundingBoxAvailable = false
        extractedFrameCount = nil
        thumbnail = nil
        isExtracting = false
        extractionProgress = 0
    }

    private func releaseSecurityScope() {
        if isAccessingSecurityScope, let url = draft.videoFile {
            url.stopAccessingSecurityScopedResource()
            isAccessingSecurityScope = false
        }
    }
}
