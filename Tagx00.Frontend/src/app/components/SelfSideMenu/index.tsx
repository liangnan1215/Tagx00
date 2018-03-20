import React from "react";
import { Menu, Icon } from 'antd';
import { inject, observer } from "mobx-react";
import { STORE_ROUTER, STORE_USER } from "../../constants/stores";
import { UserStoreProps } from "../../stores/UserStore";
import { Link } from 'react-router-dom';
import { RouterStoreProps } from "../../router/RouterStore";
import { LocaleMessage } from "../../internationalization/components";
const { SubMenu } = Menu;

interface Props extends UserStoreProps, RouterStoreProps {

}

const routes = [
  {
    path: "/self/dashboard",
    iconName: "dashboard",
    id: "selfCenter.dashboard"
  },
  {
    path: "/self/missions",
    iconName: "tag-o",
    id: "selfCenter.myMissions.menuText"
  },
  {
    path: "/self/achievement",
    iconName: "star-o",
    id: "selfCenter.achievement"
  },
  {
    path: "/self/personalInfo",
    iconName: "info",
    id: "selfCenter.personalInfo"
  }
];



@inject(STORE_USER, STORE_ROUTER)
@observer
export class SelfSideMenu extends React.Component<Props, any> {

  get selectedRoutes() {
    const router = this.props[STORE_ROUTER];
    return router.matchedPages.map(x => x.path);
  }

  render() {
    const userStore = this.props[STORE_USER];

    return <div>
      <p>welcome, {userStore.user.username}</p>
    <Menu
      mode="inline"
      selectedKeys={this.selectedRoutes}
      style={{ height: '100%' }}
    >
      {routes.map(x => <Menu.Item key={x.path}>
        <Link to={x.path}>
          <span><Icon type={x.iconName} /><LocaleMessage id={x.id}/></span>
        </Link>
      </Menu.Item>)}
    </Menu>
    </div>;
  }

}
