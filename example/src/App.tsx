/* eslint-disable @typescript-eslint/no-var-requires */
import * as React from 'react'

import { StyleSheet, View, Text, ActivityIndicator } from 'react-native'
import {
  type Tensor,
  type TensorflowModel,
  useTensorflowModel,
} from '@kyselicaa/react-native-fast-tflite'
import {
  Camera,
  CommonResolutions,
  type Frame,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
} from 'react-native-vision-camera'
import { useResizer } from 'react-native-vision-camera-resizer'
import { createSynchronizable, scheduleOnRN } from 'react-native-worklets'

function tensorToString(tensor: Tensor): string {
  return `\n  - ${tensor.dataType} ${tensor.name}[${tensor.shape}]`
}
function modelToString(model: TensorflowModel): string {
  return (
    `TFLite Model (${model.delegate}):\n` +
    `- Inputs: ${model.inputs.map(tensorToString).join('')}\n` +
    `- Outputs: ${model.outputs.map(tensorToString).join('')}`
  )
}

export default function App(): React.ReactNode {
  const { hasPermission, canRequestPermission, requestPermission } =
    useCameraPermission()
  const device = useCameraDevice('back')

  // from https://www.kaggle.com/models/tensorflow/efficientdet/frameworks/tfLite
  const model = useTensorflowModel(require('../assets/efficientdet.tflite'))
  const actualModel = model.state === 'loaded' ? model.model : undefined

  React.useEffect(() => {
    if (actualModel == null) return
    console.log(`Model loaded! Shape:\n${modelToString(actualModel)}`)
  }, [actualModel])

  const { resizer, error: resizerError } = useResizer({
    width: 320,
    height: 320,
    channelOrder: 'rgb',
    dataType: 'uint8',
    pixelLayout: 'interleaved',
    scaleMode: 'cover',
  })
  const lastResultLog = React.useMemo(() => createSynchronizable(0), [])
  const reportDetections = React.useCallback((count: number) => {
    console.log('Detections: ' + count)
  }, [])

  const onFrame = React.useCallback(
    (frame: Frame) => {
      'worklet'
      try {
        if (actualModel == null || resizer == null) return

        const resized = resizer.resize(frame)
        try {
          const input = new Uint8Array(resized.getPixelBuffer())
          const result = actualModel.runSync([input])
          const now = performance.now()
          if (now - lastResultLog.getBlocking() >= 1000) {
            lastResultLog.setBlocking(now)
            scheduleOnRN(reportDetections, Number(result[3]?.[0] ?? 0))
          }
        } finally {
          // The pixel buffer remains valid only while its GPU frame is alive.
          resized.dispose()
        }
      } finally {
        frame.dispose()
      }
    },
    [actualModel, resizer, lastResultLog, reportDetections]
  )

  const frameOutput = useFrameOutput({
    targetResolution: CommonResolutions.VGA_4_3,
    pixelFormat: 'yuv',
    dropFramesWhileBusy: true,
    onFrame,
  })

  React.useEffect(() => {
    if (canRequestPermission) requestPermission()
  }, [canRequestPermission, requestPermission])

  return (
    <View style={styles.container}>
      {hasPermission && device != null ? (
        <Camera
          device={device}
          style={StyleSheet.absoluteFill}
          isActive={true}
          outputs={[frameOutput]}
        />
      ) : (
        <Text>No Camera available.</Text>
      )}

      {(model.state === 'loading' ||
        (resizer == null && resizerError == null)) && (
        <ActivityIndicator size="small" color="white" />
      )}

      {model.state === 'error' && (
        <Text>Failed to load model! {model.error.message}</Text>
      )}
      {resizerError != null && (
        <Text>Failed to create resizer! {resizerError.message}</Text>
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
})
