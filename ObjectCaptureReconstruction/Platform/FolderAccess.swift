/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Platform differences in reaching user-selected folders: bookmark options,
security-scoped access lifetime, and revealing finished models to the user.
*/

import Foundation

#if os(macOS)
import AppKit
#endif

/// Bookmark options for persisting access to user-selected folders.
///
/// macOS needs an explicit security scope on both creation and resolution.
/// Bookmarks created by an iPhone or iPad app are implicitly security scoped,
/// so the options stay empty there.
enum FolderBookmark {
    static var creationOptions: URL.BookmarkCreationOptions {
        #if os(macOS)
        return [.withSecurityScope]
        #else
        return []
        #endif
    }

    static var resolutionOptions: URL.BookmarkResolutionOptions {
        #if os(macOS)
        return [.withSecurityScope]
        #else
        return []
        #endif
    }
}

/// Holds security-scoped access to a URL for as long as the instance lives.
///
/// Use it when access has to outlive a single function, such as while a share
/// sheet reads exported models on iPhone and iPad.
final class SecurityScopedAccess {
    private let url: URL?

    init(_ url: URL) {
        self.url = url.startAccessingSecurityScopedResource() ? url : nil
    }

    deinit {
        url?.stopAccessingSecurityScopedResource()
    }
}

#if os(macOS)
/// Shows finished output files in the Finder. iPhone and iPad share finished
/// models through the system share sheet instead.
enum OutputReveal {
    /// Selects `files` in the Finder, or opens `folder` when no file exists yet.
    static func reveal(files: [URL], inFolder folder: URL) {
        if files.isEmpty {
            NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: folder.path)
        } else {
            NSWorkspace.shared.activateFileViewerSelecting(files)
        }
    }
}
#endif
