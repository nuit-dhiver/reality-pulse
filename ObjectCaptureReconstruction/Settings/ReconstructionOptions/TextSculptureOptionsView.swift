/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Options for the text-sculpture export: layout, letter size, the words to write
(typed or loaded from a text file), and texture versus single-ink coloring.
*/

import SwiftUI
import UniformTypeIdentifiers

struct TextSculptureOptionsView: View {
    @Environment(JobDraft.self) private var draft: JobDraft
    @State private var showFileImporter = false

    var body: some View {
        @Bindable var draft = draft

        VStack(alignment: .leading, spacing: 8) {
            Picker("Layout:", selection: $draft.textSculptureOptions.layout) {
                ForEach(TextSculptureOptions.Layout.allCases, id: \.self) { layout in
                    Text(layout.displayName)
                }
            }
            .pickerStyle(.segmented)

            Text(layoutDescription)
                .font(.caption2)
                .foregroundStyle(.tertiary)

            Picker("Letter size:", selection: $draft.textSculptureOptions.letterSize) {
                ForEach(TextSculptureOptions.LetterSize.allCases, id: \.self) { size in
                    Text(size.displayName)
                }
            }
            .pickerStyle(.segmented)

            ZStack(alignment: .topLeading) {
                TextEditor(text: $draft.textSculptureOptions.text)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(4)
                    .frame(height: 80)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(6)

                if draft.textSculptureOptions.text.isEmpty {
                    Text("Words to write. Leave empty to use the model name.")
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 4)
                        .allowsHitTesting(false)
                }
            }

            Button("Load from File…") {
                showFileImporter = true
            }

            Toggle("Single ink color", isOn: $draft.textSculptureOptions.useInkColor)

            if draft.textSculptureOptions.useInkColor {
                ColorPicker("Ink:", selection: inkColor, supportsOpacity: false)
            }
        }
        .fileImporter(isPresented: $showFileImporter, allowedContentTypes: [.plainText]) { result in
            switch result {
            case .success(let url):
                loadText(from: url)
            case .failure(let error):
                draft.alertMessage = "\(error)"
                draft.hasError = true
            }
        }
    }

    private var layoutDescription: String {
        switch draft.textSculptureOptions.layout {
        case .rings:
            return "Readable lines of text wrap around the model."
        case .cloud:
            return "One character per surface point, like a point cloud of letters."
        }
    }

    private var inkColor: Binding<Color> {
        Binding(
            get: {
                let ink = draft.textSculptureOptions.inkColor
                return Color(.sRGB, red: Double(ink.x), green: Double(ink.y), blue: Double(ink.z))
            },
            set: { color in
                guard let srgb = NSColor(color).usingColorSpace(.sRGB) else { return }
                draft.textSculptureOptions.inkColor = SIMD3<Float>(
                    Float(srgb.redComponent),
                    Float(srgb.greenComponent),
                    Float(srgb.blueComponent)
                )
            }
        )
    }

    /// Copy the file's contents into the job, so no file access is needed later.
    private func loadText(from url: URL) {
        let gotAccess = url.startAccessingSecurityScopedResource()
        defer { if gotAccess { url.stopAccessingSecurityScopedResource() } }

        do {
            if let utf8 = try? String(contentsOf: url, encoding: .utf8) {
                draft.textSculptureOptions.text = utf8
            } else {
                var encoding = String.Encoding.utf8
                draft.textSculptureOptions.text = try String(contentsOf: url, usedEncoding: &encoding)
            }
        } catch {
            draft.alertMessage = "Couldn't read \(url.lastPathComponent): \(error.localizedDescription)"
            draft.hasError = true
        }
    }
}
