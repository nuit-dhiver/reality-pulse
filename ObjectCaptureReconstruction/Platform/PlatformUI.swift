/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
SwiftUI shims for the controls that differ between the macOS build and the
iPhone and iPad builds.
*/

import SwiftUI

#if os(macOS)
import AppKit
import UniformTypeIdentifiers
#else
import UIKit
#endif

/// A folder icon: the Finder folder icon on macOS, a symbol on iPhone and iPad.
struct PlatformFolderIcon: View {
    var body: some View {
        #if os(macOS)
        Image(nsImage: NSWorkspace.shared.icon(for: .folder))
            .resizable()
            .aspectRatio(contentMode: .fit)
        #else
        // Symbols keep their natural size next to the folder name.
        Image(systemName: "folder.fill")
            .foregroundStyle(.tint)
        #endif
    }
}

/// The icon of a specific file: the Finder icon on macOS, `symbolName` on
/// iPhone and iPad, where apps cannot read Finder document icons.
struct PlatformFileIcon: View {
    let url: URL
    let symbolName: String

    var body: some View {
        #if os(macOS)
        Image(nsImage: NSWorkspace.shared.icon(forFile: url.path))
            .resizable()
            .aspectRatio(contentMode: .fit)
        #else
        Image(systemName: symbolName)
            .foregroundStyle(.tint)
        #endif
    }
}

#if !os(macOS)
/// The system share sheet, used on iPhone and iPad in place of revealing
/// finished models in the Finder.
struct ShareSheet: UIViewControllerRepresentable {
    let urls: [URL]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: urls, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
#endif

extension View {
    /// Checkbox toggles on macOS, and the platform default on iPhone and iPad,
    /// where `CheckboxToggleStyle` is unavailable.
    @ViewBuilder
    func platformCheckboxToggleStyle() -> some View {
        #if os(macOS)
        toggleStyle(.checkbox)
        #else
        self
        #endif
    }

    /// The queue list style: alternating row backgrounds on macOS, which iOS
    /// lists do not support.
    @ViewBuilder
    func platformQueueListStyle() -> some View {
        #if os(macOS)
        listStyle(.inset(alternatesRowBackgrounds: true))
        #else
        listStyle(.inset)
        #endif
    }

    /// A minimum sheet size on macOS. iPhone and iPad size sheets themselves,
    /// so a minimum would only force content off screen.
    @ViewBuilder
    func platformSheetFrame(minWidth: CGFloat, minHeight: CGFloat) -> some View {
        #if os(macOS)
        frame(minWidth: minWidth, minHeight: minHeight)
        #else
        self
        #endif
    }
}
