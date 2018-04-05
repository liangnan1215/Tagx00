import React from "react";
import { List } from 'antd';
import { MyMissionCard } from "./MyMissionCard";
import { Instance } from "../../models/instance/Instance";

interface Props {
  items: Instance[];

}

export class MissionCardPane extends React.Component<Props, any> {
  render() {
    return <List
      grid={{ gutter: 16, xs: 1, sm: 2, xl: 3, xxl: 4 }}
      dataSource={this.props.items}
      renderItem={(item: Instance) => (
        <List.Item key={item.instanceId}>
          <MyMissionCard instance={item}/>
        </List.Item>
      )}
    />
  }
}
