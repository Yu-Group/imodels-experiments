
// console.clear();



/**********************************************
 📸️ Image Vars
 *********************************************/
// collection of emoji urls
const emojis = [
// "https://assets.codepen.io/t-1043/happytear.png",
// "https://assets.codepen.io/t-1043/heh.png",
// "https://assets.codepen.io/t-1043/hellzya.png",
// "https://assets.codepen.io/t-1043/hmph.png",
// "https://assets.codepen.io/t-1043/okei.png",
//     "https://demos.csinva.io/logo_figs.svg",
    "logo_figs.svg",
    "logo_figs (1).svg",
    "logo_figs (2).svg",
  // "https://csinva.io/imodels/img/model_table.png",
  //   "https://csinva.io/imodels/img/imodels_logo.svg",
// "https://assets.codepen.io/t-1043/whew.png"
];


// preloaded p5 images, ready to be drawn
const preloadedImages = [];
// fetch a random (preloaded) emoji
const randomImage = preloadedImages[Math.floor(Math.random() * preloadedImages.length)];




/**********************************************
 🖌️Canvas Vars
 *********************************************/
let vpWidth = window.innerWidth;
let vpHeight = window.innerHeight;
let ground, ceiling, leftWall, rightWall;
// collection of containers: ground, ceiling, leftWall, right
let barriers = [];
// boxes = emojis
let boxes = [];
// maximum emojis
const maxBoxes = 175;




/**********************************************
 🖥️ Interface & Interaction
 *********************************************/
const instruction = document.querySelector(".instruction");
const reset = document.querySelector(".js-reset");
const count = document.querySelector(".js-count");
// const drawnEmojis = document.querySelector(".js-total");
let instructionTimeout;
let numOfDrags = 0;
// drawnEmojis.innerHTML = maxBoxes;
// reset.addEventListener("click", resetApp);
window.addEventListener("resize", () => {
  vpWidth = window.innerWidth;
});





/**********************************************
 🎈️Physics Engine
 *********************************************/
const { Engine, World, Bodies, Mouse, MouseConstraint, Constraint } = Matter;
let world, engine;
let mConstraint;




/**********************************************
 📖️Some Handy Classes
 *********************************************/

/* Box: Used to create barriers */
class Box {
  constructor(x, y, w, h) {
    this.body = Matter.Bodies.rectangle(x, y, w, h);
    this.body.restitution = 0.8;
    this.body.friction = 0;
    Matter.World.add(world, this.body);
    this.w = w;
    this.h = h;
    this.x = x;
    this.y = y;
    this.image = randomImage;
  }

  show() {
    const pos = this.body.position;
    const angle = this.body.angle;
    push();
    translate(pos.x, pos.y);
    rotate(angle);
    fill(255);
    rectMode(CENTER);
    imageMode(CENTER);
    image(this.image, 0, 0, this.w, this.h);

    pop();
  }}


/* Emoji: Used to create create emojis */
class Emoji {
  constructor(x, y, r) {
    const randomX = (Math.random() - 0.5) * 1;
    const randomY = (Math.random() - 0.5) * 1;
    // this.body = Matter.Bodies.circle(x, y, r);
    this.body = Matter.Bodies.rectangle(x, y, r * 2, r * 2);
    this.body.restitution = 0.8;
    this.body.friction = 0;
    Matter.Body.applyForce(
    this.body,
    { x: x, y: y },
    { x: randomX, y: randomY });

    Matter.Body.setMass(this.body, this.body.mass * 4);
    Matter.World.add(world, this.body);
    this.r = r;
    this.background =
    preloadedImages[Math.floor(Math.random() * preloadedImages.length)];
  }
  show() {
    const pos = this.body.position;
    const angle = this.body.angle;
    push();
    translate(pos.x, pos.y);
    rotate(angle);
    imageMode(CENTER);
    image(this.background, 0, 0, this.r * 2, this.r * 2);
    pop();
  }}


/* Ground: Barrier */
class Ground extends Box {
  constructor(x, y, w, h) {
    super(x, y, w, h);
    this.body.isStatic = true;
    this.body.restitution = 1;
    this.body.friction = 0;
  }

  show() {
    const pos = this.body.position;
    const angle = this.body.angle;
    push();
    translate(pos.x, pos.y);
    rotate(angle);
    noStroke();
    fill(30);
    rectMode(CENTER);
    rect(0, 0, this.w, this.h);
    pop();
  }}





/**********************************************
 😄️ Helper Functions
 *********************************************/

// function resetApp() {
//   boxes.length = 0;
//   // update count
//   count.innerHTML = boxes.length;
//   setup();
// };

function resetBarrier() {
  barriers.length = 0;
}

function createBarriers() {
  barrierWidth = 50;
  barrierOffset = barrierWidth * 0.5;
  ground = new Ground(width / 2, height + barrierOffset, width, barrierWidth);
  ceiling = new Ground(width / 2, -barrierOffset, width, barrierWidth);
  leftWall = new Ground(-barrierOffset, height / 2, barrierWidth, height);
  rightWall = new Ground(
  width + barrierOffset,
  height / 2,
  barrierWidth,
  height);


  barriers.push(ground, ceiling, leftWall, rightWall);
}

function isOutOfView(el) {
  let x = el.body.position.x;
  let y = el.body.position.y;

  if (x < -100 || x > windowWidth + 100 || y < -100 || y > windowHeight + 100) {
    // count.innerHTML = boxes.length;
    return true;
  } else {
    return false;
  }
}

function timeoutInstruction() {
  clearTimeout(instructionTimeout);
  instructionTimeout = setTimeout(() => {
    hideInstruction();
  }, 2000);
}




/**********************************************
 🤓️ P5!
 *********************************************/

function preloadImages() {
  // convert URL to p5 images
  emojis.forEach(emoji => {
    // load image
    const image = loadImage(emoji);
    // save for later
    preloadedImages.push(image);
  });
}

function setup() {
  const canvas = createCanvas(windowWidth, windowHeight * 0.4);
  canvas.parent("p5");
  vpHeight = window.innerHeight;
  vpWidth = window.innerWidth;
  engine = Engine.create();
  world = engine.world;
  engine.timing.timeScale = 0.4; //0.8;

  // setup barriers
  if (barriers.length > 0) resetBarrier();
  createBarriers();

  // load images
  preloadImages();

  // draw boxes if they already exist (resized window)
  for (let i = 0; i < boxes.length; i++) {
    Matter.World.add(world, boxes[i].body);
  }

  const mouse = Mouse.create(canvas.elt);
  const options = {
    mouse: mouse };


  // A fix for HiDPI displays
  mouse.pixelRatio = pixelDensity();
  mConstraint = MouseConstraint.create(engine, options);
  mConstraint.constraint.stiffness = 0.7;
  mConstraint.constraint.damping = 0.2;
  World.add(world, mConstraint);
}

function mousePressed() {
  numOfDrags++;


  const divider = 40;
  const diameter = vpWidth > vpHeight ? vpWidth / divider : vpHeight / divider;

  const newEl = new Emoji(mouseX, mouseY, diameter, diameter);
  boxes.push(newEl);
}

function mouseDragged() {
  const divider = 40;
  const diameter = vpWidth > vpHeight ? vpWidth / divider : vpHeight / divider;

  const newEl = new Emoji(mouseX, mouseY, diameter, diameter);
  boxes.push(newEl);
}

/// create something every second
var intervalId = window.setInterval(function(){
  const divider = 40;
  const diameter = vpWidth > vpHeight ? vpWidth / divider : vpHeight / divider;
  const randomX = (Math.random()) * vpWidth;
  const randomY = (Math.random()) * vpHeight * 0.3;
  const newEl = new Emoji(randomX, randomY, diameter, diameter);
  boxes.push(newEl);
  /// call your function here
}, 1000);

function draw() {
  background(255);
  Matter.Engine.update(engine);

  for (let barrier of barriers) {
    barrier.show();
  }

  // draw only the boxes in frame
  // delete the rest
  for (let i = 0; i < boxes.length; i++) {
    boxes[i].show();

    if (isOutOfView(boxes[i])) {
      Matter.World.remove(world, boxes[i].body);
      boxes.splice(i, 1);
      i--;
    }
  }

  if (boxes.length > maxBoxes) {
    Matter.World.remove(world, boxes[0].body);
    boxes.splice(0, 1);
  }

  // update count
  // count.innerHTML = boxes.length;
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  setup();
}