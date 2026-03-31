/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Configuration UI for Structure from Motion options.
*/

import SwiftUI

struct SfMOptionsView: View {
    @Binding var configuration: CodableSfMConfiguration

    var body: some View {
        Section("SfM Settings") {
            qualityPresetPicker
            matcherTypePicker
            cameraModelPicker
            sharedIntrinsicsToggle

            if configuration.qualityPreset == .detailed {
                maxFeaturesField
            }

            gpuToggle
        }
    }

    // MARK: - Subviews

    private var qualityPresetPicker: some View {
        Picker("Quality", selection: $configuration.qualityPreset) {
            ForEach(SfMQualityPreset.allCases) { preset in
                Text(preset.displayName).tag(preset)
            }
        }
        .help("Controls the number of features extracted and matching thoroughness.")
    }

    private var matcherTypePicker: some View {
        Picker("Matcher", selection: $configuration.matcherType) {
            ForEach(SfMMatcherType.allCases) { matcher in
                Text(matcher.displayName).tag(matcher)
            }
        }
        .help("Exhaustive: match all pairs (unordered images). Sequential: match nearby frames (video).")
    }

    private var cameraModelPicker: some View {
        Picker("Camera Model", selection: $configuration.cameraModel) {
            ForEach(COLMAPCameraModel.allCases) { model in
                Text(model.displayName).tag(model)
            }
        }
        .help("Lens distortion model for intrinsic estimation. OpenCV works for most cameras.")
    }

    private var sharedIntrinsicsToggle: some View {
        Toggle("Shared Camera Intrinsics", isOn: $configuration.sharedIntrinsics)
            .help("Enable if all images were taken with the same camera and settings.")
    }

    private var maxFeaturesField: some View {
        HStack {
            Text("Max Features")
            Spacer()
            TextField("Override", value: $configuration.maxNumFeaturesOverride, format: .number)
                .frame(width: 80)
                .textFieldStyle(.roundedBorder)
            Text("(0 = auto)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .help("Override the maximum number of SIFT features per image. 0 uses the preset default.")
    }

    private var gpuToggle: some View {
        Toggle("Use GPU Acceleration", isOn: $configuration.useGPU)
            .help("Use GPU for feature extraction and matching when available.")
    }
}
