module.exports = {
  presets: [
    ['@babel/preset-env', { targets: { node: 'current' } }],
    '@babel/preset-react',
    '@babel/preset-typescript',
    // Strip Flow types from react-native and @react-native packages
    ['@babel/preset-flow', { all: true }],
  ],
};
