import React from 'react';
import { Card, Button, Icon } from 'antd';
import { LocaleMessage } from "../internationalization/components";


interface Props {
  goNext: () => void;
  goPrevious: () => void;
  previousAvailable: boolean;
  readonlyMode: boolean;
  saveProgress: () => void;
}

export class ProgressController extends React.Component<Props, {}> {
  render() {
    const prefix = "drawingPad.common.progressController.";
    return <Card style={{marginTop: "8px"}}>
      <Button.Group size={"large"}>
        <Button onClick={this.props.goPrevious}>
          <Icon type="left" /><LocaleMessage id={prefix+"backward"}/>
        </Button>
        {this.props.readonlyMode ? null :
          <Button type="primary" onClick={this.props.saveProgress}>
            <Icon type="save"/><LocaleMessage id={prefix + "save"}/>
          </Button>
        }
        <Button onClick={this.props.goNext}>
          <LocaleMessage id={prefix+"forward"}/><Icon type="right" />
        </Button>
      </Button.Group>
    </Card>;
  }
}
