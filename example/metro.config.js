const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config')
const path = require('path')
const pak = require('../package.json')

const root = path.resolve(__dirname, '..')
const modules = Object.keys(pak.peerDependencies)
const defaultConfig = getDefaultConfig(__dirname)

/**
 * @type {import('metro-config').MetroConfig}
 */
const config = {
  watchFolders: [root],

  // Library source and the example must resolve the same React/native modules.
  resolver: {
    assetExts: [...defaultConfig.resolver.assetExts, 'tflite'],
    disableHierarchicalLookup: true,
    nodeModulesPaths: [
      path.join(__dirname, 'node_modules'),
      path.join(root, 'node_modules'),
    ],
    extraNodeModules: modules.reduce((acc, name) => {
      acc[name] = path.join(__dirname, 'node_modules', name)
      return acc
    }, {}),
  },

  transformer: {
    getTransformOptions: async () => ({
      transform: {
        experimentalImportSupport: false,
        inlineRequires: true,
      },
    }),
  },
}

module.exports = mergeConfig(defaultConfig, config)
