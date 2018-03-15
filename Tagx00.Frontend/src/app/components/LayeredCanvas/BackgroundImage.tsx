import React, {CSSProperties} from "react";
import {action, observable} from "mobx";
import {observer} from "mobx-react";

interface BackgroundImageProps {
  imageUrl: string;
  onLoad: (width: number, height: number) => void;
  initialHeight: number;
  initialWidth: number;
}

export class BackgroundImage extends React.Component<BackgroundImageProps, {}> {

  onLoad = ({target}) => {
    this.props.onLoad(target.width, target.height);
  };

  render() {
    const style: CSSProperties = {
      position: "absolute"
    };
    return <img style={style} src={this.props.imageUrl} onLoad={this.onLoad}/>;

  }
}