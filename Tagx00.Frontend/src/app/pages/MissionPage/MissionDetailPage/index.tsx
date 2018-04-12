import React from 'react';
import { Inject } from "react.di";
import { MissionService } from "../../../api/MissionService";
import { UserStore } from "../../../stores/UserStore";
import { AsyncComponent } from "../../../router/AsyncComponent";
import { Loading } from "../../../components/Common/Loading";

interface Props {
  missionId: string;
}

export class MissionDetailPage extends React.Component<Props, {}> {


  @Inject missionService: MissionService;
  @Inject userStore: UserStore;

  renderPage = async () => {
    const detail = await this.missionService.getAMission(this.props.missionId, this.userStore.token);
    return JSON.stringify(detail);
  };

  render() {
    return <AsyncComponent render={this.renderPage} componentWhenLoading={<Loading/>}/>;
  }
}
