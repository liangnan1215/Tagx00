import React from "react";


export function removeElementAt<T>(array: Array<T>, index: number) {
  array.splice(index, 1);
}

export function replaceElement<T>(array: Array<T>, source: T, replacement: T) {
  const index = array.indexOf(this.source);
  array[index] = replacement;
}

export function flatten<T>(nestedArray: Array<Array<T>>) {
  return nestedArray.reduce((prev, curr) => [...prev, ...curr], []);
}

export function takeAtMost<T>(array: Array<T>, n: number) {
  if (array.length<=n) {
    return array;
  } else {
    return array.slice(0,n);
  }
}
