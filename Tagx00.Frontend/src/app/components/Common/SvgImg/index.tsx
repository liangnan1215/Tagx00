import * as React from "react";

interface Props {
  filePath: string;
  height: number;
  width: number;
}

export class SvgImg extends React.Component<Props, {}> {

  render() {
    const Svg = require(`svg-react-loader?name=Svg!../../../../assets/svg/${this.props.filePath}`);
    return <Svg width={this.props.width} height={this.props.height}/>;
  }

}
