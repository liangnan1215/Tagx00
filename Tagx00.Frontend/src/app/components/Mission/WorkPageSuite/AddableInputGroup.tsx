import React, { ReactNode } from "react";
import { Button, Icon, Input } from 'antd';

interface Props {
  items: string[];
  onChange: (newItems: string[]) => void;
  inputPrompt?: string;
  addingButtonPlaceholder?: ReactNode;
  readonly: boolean;
}

export class AddableInputGroup extends React.PureComponent<Props, any> {


  onInputChange = (newValue: string, index: number) => {
    const items = this.props.items.slice();
    items[index] = newValue;
    this.props.onChange(items);

  };

  addADescription = () => {
    this.props.onChange(this.props.items.concat([""]));
  };

  removeOne = (index: number) => {
    const items = this.props.items.slice();
    items.splice(index,1);
    this.props.onChange(items);
  };

  render() {
    return <>
      {this.props.items.map((x, index) =>
        <Input placeholder={this.props.inputPrompt}
               value={x}
               key={index}
               disabled={this.props.readonly}
               onChange={(e) => this.onInputChange(e.target.value, index)}
               addonAfter={this.props.readonly ? null : <Icon type="minus-circle-o" onClick={() => this.removeOne(index)}/>}
        />)
      }
      {this.props.readonly ? null :
        <Button type="dashed" onClick={this.addADescription}>
          {this.props.addingButtonPlaceholder || null}
        </Button>
      }
    </>;

  }
}
