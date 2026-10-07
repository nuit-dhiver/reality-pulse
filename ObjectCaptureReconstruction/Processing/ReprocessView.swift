/*
See the LICENSE file for licensing information.

Abstract:
Show a reprocessing option.
*/

import SwiftUI

struct ReprocessView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel

    var body: some View {
        HStack {
            Text("All models completed.")

            Spacer()

            Button("Reprocess...") {
                appDataModel.state = .ready
            }
        }
        .padding(.horizontal)
        .padding(.top, 8)
        .padding(.bottom, 16)
    }
}
