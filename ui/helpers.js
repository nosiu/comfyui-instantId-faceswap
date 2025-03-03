export const getPointsCenter = (points) => {
  let sumX = 0, sumY = 0, sumZ = 0;
  points.forEach(([x, y, z]) => {
      sumX += x;
      sumY += y;
      if (z != null) sumZ += z
  });

  const ret = [sumX / points.length, sumY / points.length]
  if (points[0].length > 2) ret.push(sumZ / points.length)
  return ret
}

export const getPoinsMinMax = (points) => {
  let minX = points[0][0], maxX = points[0][0];
  let minY = points[0][1], maxY = points[0][1];
  points.forEach(([x, y]) => {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
  });
  return { minX, maxX, minY, maxY };
}

export const drawKps = (canvas, kps, opacities) => {
  const color_list = [
    `255, 0, 0,`,
    `0, 255, 0,`,
    `0, 0, 255,`,
    `255, 255, 0,`,
    `255, 0, 255,`
  ]

    const ctx = canvas.getContext("2d")
  const stickWidth = 10;
  const limbSeq = [[0, 2], [1, 2], [3, 2], [4, 2]]

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = "black"
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.save()
  limbSeq.forEach((limb, idx) => {
    const kp1 = kps[limb[0]]
    const kp2 = kps[limb[1]]
    const color = `rgba( ${color_list[limb[0]]} ${0.6 * opacities[limb[0]]})`

    const x = [kp1[0], kp2[0]];
    const y = [kp1[1], kp2[1]];
    const length = Math.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
    const angle = Math.atan2(y[1] - y[0], x[1] - x[0])

    const num_points = 20;
    const polygon = []

    const midX = (x[0] + x[1]) / 2
    const midY = (y[0] + y[1]) / 2

    for (let i = 0; i <= num_points; i++) {
      const theta = (i / num_points) * Math.PI * 2
      const dx = (length / 2) * Math.cos(theta);
      const dy = (stickWidth / 2) * Math.sin(theta);
      const rx = Math.cos(angle) * dx - Math.sin(angle) * dy + midX
      const ry = Math.sin(angle) * dx + Math.cos(angle) * dy + midY
      polygon.push([rx, ry]);
    }

    ctx.beginPath();
    ctx.moveTo(polygon[0][0], polygon[0][1])
    for (let i = 1; i < polygon.length; i++) {
      ctx.lineTo(polygon[i][0], polygon[i][1])
    }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  })

  kps.forEach((kp, idx) => {
    const [x, y] = kp;
    const color = `rgba( ${color_list[idx]} ${opacities[idx]})`
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  });
  ctx.restore();
}

export const checkWebGlSupport = () => {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2")
  return !!gl
}

export const normalizePoints = (points, w, h) => {
  const minValues = [
    Math.min(...points.map(p => p[0])),
    Math.min(...points.map(p => p[1])),
    Math.min(...points.map(p => p[2]))
  ];

  const maxValues = [
    Math.max(...points.map(p => p[0])),
    Math.max(...points.map(p => p[1])),
    Math.max(...points.map(p => p[2]))
  ];

  const ranges = [
    maxValues[0] - minValues[0], 
    maxValues[1] - minValues[1],
    maxValues[2] - minValues[2] 
  ];

  const scaleX = w / ranges[0]
  const scaleY = h / ranges[1]

  const scaleFactor = Math.min(scaleX, scaleY);

  const normalizedPoints = points.map(point => [
    (point[0] - minValues[0]) * scaleFactor,
    (point[1] - minValues[1]) * scaleFactor,
    (point[2] - minValues[2]) * scaleFactor
  ]);

  const maxNormalizedValues = [
    Math.max(...normalizedPoints.map(p => p[0])),
    Math.max(...normalizedPoints.map(p => p[1])),
    Math.max(...normalizedPoints.map(p => p[2]))
  ];

  const centerOffset = [
    (w - maxNormalizedValues[0]) / 2,
    (h - maxNormalizedValues[1]) / 2,
    -maxNormalizedValues[2] / 2
  ];

  const centeredPoints = normalizedPoints.map(point => [
    point[0] + centerOffset[0],
    point[1] + centerOffset[1],
    point[2] + centerOffset[2]
  ]);

  return centeredPoints;
}

export const rotatePoints3D = (points, kps, angleXDeg, angleYDeg, angleZDeg) => {
  const angleX = angleXDeg * (Math.PI / 180);
  const angleY = angleYDeg * (Math.PI / 180);
  const angleZ = angleZDeg * (Math.PI / 180);

  const numPoints = kps.length;
  const center = kps.reduce((acc, point) => {
    acc[0] += point[0]
    acc[1] += point[1]
    acc[2] += point[2]
    return acc;
  }, [0, 0, 0]).map(coord => coord / numPoints)

  const translatedPoints = points.map(point => [
    point[0] - center[0],
    point[1] - center[1],
    point[2] - center[2]
  ]);

  function rotateX(point, angle) {
    const cosTheta = Math.cos(angle);
    const sinTheta = Math.sin(angle);
    return [
      point[0],
      point[1] * cosTheta - point[2] * sinTheta,
      point[1] * sinTheta + point[2] * cosTheta
    ];
  }

  function rotateY(point, angle) {
    const cosTheta = Math.cos(angle);
    const sinTheta = Math.sin(angle);
    return [
      point[0] * cosTheta + point[2] * sinTheta,
      point[1],
      -point[0] * sinTheta + point[2] * cosTheta
    ];
  }

  function rotateZ(point, angle) {
    const cosTheta = Math.cos(angle);
    const sinTheta = Math.sin(angle);
    return [
      point[0] * cosTheta - point[1] * sinTheta,
      point[0] * sinTheta + point[1] * cosTheta,
      point[2]
    ];
  }

  const rotatedPoints = translatedPoints.map(point => {
    let rotatedPoint = rotateX(point, angleX)
    rotatedPoint = rotateY(rotatedPoint, angleY)
    rotatedPoint = rotateZ(rotatedPoint, angleZ)
    return rotatedPoint;
  });

  const finalPoints = rotatedPoints.map(point => [
    point[0] + center[0],
    point[1] + center[1],
    point[2] + center[2]
  ])

  return finalPoints
}

export const getImgFromInput = (inputNode) => {
  if (inputNode?.type === "Reroute") {
    return getImgFromInput(inputNode.getInputNode(0))
  }
  return inputNode
} 

export const getDefaultKpsData = () => ({
    array: [], height: 0, width: 0, rotateX: 0, rotateY: 0, rotateZ: 0 , opacities: [1, 1, 1, 1, 1],
    jawline: [], eyebrow_left: [], eyebrow_right: [], nose_bridge: [], nose_lower: [], 
    eye_left: [], eye_right: [], mouth_outer: [], mouth_inner: []
})