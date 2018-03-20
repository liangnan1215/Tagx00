import { UserStoreProps } from "../../../stores/UserStore";
import React from "react";
import { STORE_USER } from "../../../constants/stores";
import { Dropdown, Icon, Menu } from 'antd';
import { inject, observer } from "mobx-react";
import { LocaleMessage } from "../../../internationalization/components";
import { Link } from "react-router-dom";

interface Props extends UserStoreProps {

}

@inject(STORE_USER)
@observer
export class UserIndicator extends React.Component<Props, {}> {

  logout = () => {
    this.props[STORE_USER].logout();
  };

  render() {
    const userStore = this.props[STORE_USER];
    const dropdownMenu = <Menu>
      <Menu.Item key="self">
        <Link to={"/self"}><LocaleMessage id={"navbar.selfCenter"}/></Link>
      </Menu.Item>
      <Menu.Divider />
      <Menu.Item key="logout">
        <a onClick={this.logout}><LocaleMessage id={"navbar.logout"}/></a>
      </Menu.Item>
    </Menu>;

    return <Dropdown overlay={dropdownMenu} trigger={["click"]}>
      <a className="ant-dropdown-link">
        <Icon type="user"/> <LocaleMessage id={"navbar.welcome"} replacements={{
        username: userStore.user.username
      }}/> <Icon type="down"/>
      </a></Dropdown>;
  }
}
