import fs from 'fs'
import path from 'path'
import {
  ConfigPlugin,
  withPlugins,
  withDangerousMod,
  withXcodeProject,
  ExportedConfigWithProps,
  XcodeProject,
} from '@expo/config-plugins'
import { insertContentsAtOffset } from '@expo/config-plugins/build/utils/commonCodeMod'

export const withMetalDelegate: ConfigPlugin = (config) =>
  withPlugins(config, [
    // Add $EnableMetalDelegate = true to the top of the Podfile
    [
      withDangerousMod,
      [
        'ios',
        (currentConfig: ExportedConfigWithProps<unknown>) => {
          const podfilePath = path.join(
            currentConfig.modRequest.platformProjectRoot,
            'Podfile'
          )

          const prevPodfileContent = fs.readFileSync(podfilePath, 'utf-8')
          const newPodfileContent = insertContentsAtOffset(
            prevPodfileContent,
            '$EnableMetalDelegate=true\n',
            0
          )

          fs.writeFileSync(podfilePath, newPodfileContent)

          return currentConfig
        },
      ],
    ],
    // Add the Metal framework to the Xcode project
    [
      withXcodeProject,
      (currentConfig: ExportedConfigWithProps<XcodeProject>) => {
        // `XcodeProject` isn't correctly exported from @expo/config-plugins, so TypeScript sees it as `any`
        // see https://github.com/expo/expo/issues/16470
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        const xcodeProject = currentConfig.modResults
        xcodeProject.addFramework('Metal.framework')

        return currentConfig
      },
    ],
  ])
