import React from "react"
import { Icon, Button } from "antd";
import { Localize } from "../../internationalization/components";

export class RegisterSuccessShower extends React.Component<any, any> {
  render() {
    return (
      <Localize replacements={{}}>{
        (props) => {
          return (
            <div style={{textAlign: 'center'}}>
              <Icon type="check-circle-o" style={{fontSize: 200, color: "green"}}/>
              <div style={{padding: "30%", paddingTop: "10%"}}>
                <h3>注册成功</h3>
                <div style={{margin: "5%"}}>
                  <Button type="primary" size="large">返回首页</Button>
                </div>
              </div>
            </div>);
        }
      }
      </Localize>)
  }
}