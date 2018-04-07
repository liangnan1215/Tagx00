import { Point } from "../../../../models/instance/image/Shapes";

export class Rectangle {
  id: number;
  start: Point = {x: 0, y: 0};
  end: Point = {x: 0, y: 0};

  constructor(params: Partial<Rectangle> = {}) {
    Object.assign(this, params);
  }

  get width() {
    return Math.abs(this.end.x - this.start.x);
  }

  get height() {
    return Math.abs(this.end.y - this.start.y);
  }

  get x() {
    return Math.min(this.start.x, this.end.x);
  }

  get y() {
    return Math.min(this.start.y, this.end.y);
  }

  get leftTop() {
    return {x: this.x, y: this.y};
  }

  get rightBottom() {
    return {x: this.x + this.width, y: this.y + this.height};
  }

  isOnSides(point: Point) {
    const error = 5;
    const leftUpper = {
      x: this.x, y: this.y
    };
    const rightUpper = {
      x: this.x + this.width, y: this.y
    };
    const leftDown = {
      x: this.x, y: this.y + this.height
    };

    const rightDown = {
      x: this.x + this.width, y: this.y + this.height
    };

    return leftUpper.y <= point.y && point.y <= leftDown.y
      && leftUpper.x <= point.x && point.x <= rightUpper.x;

    // return (Math.abs(point.y - leftUpper.y) <= error && point.x - leftUpper.x <= this.width)
    //   || (Math.abs(point.y - leftDown.y) <= error && point.x - leftDown.x <= this.width)
    //   || (Math.abs(point.x - leftUpper.x) <= error && point.y - leftUpper.y <= this.height)
    //   || (Math.abs(point.x - rightDown.x) <= error && point.y - rightUpper.y <= this.height);

  }
}
