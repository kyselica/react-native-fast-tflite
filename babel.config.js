module.exports = function (api) {
  if (api.env('test') === true)
    return { presets: ['module:@react-native/babel-preset'] }
  return {
    presets: [
      ['@babel/preset-env', { targets: { node: 'current' } }],
      '@babel/preset-react',
      '@babel/preset-typescript',
      ['@babel/preset-flow', { all: true }],
    ],
  }
}
