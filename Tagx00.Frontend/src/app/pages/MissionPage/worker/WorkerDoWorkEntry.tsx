import React from 'react';
import { WorkerService } from "../../../api/WorkerService";
import { Inject } from "react.di";
import { UserStore } from "../../../stores/UserStore";
import { MissionService } from "../../../api/MissionService";
import { DoWorkPage } from "../DoWorkPage";
import { AsyncComponent } from "../../../router/AsyncComponent";
import { Loading } from "../../../components/Common/Loading";

interface Props {
  missionId: string;
}

export class WorkerDoWorkEntry extends React.Component<Props, {}> {
  @Inject workerService: WorkerService;
  @Inject missionService: MissionService;
  @Inject userStore: UserStore;

  renderContent = async () => {
    const token = this.userStore.token;
    const instanceDetail = await this.workerService.getInstanceDetail(this.props.missionId, token);
    const missionDetail = await this.missionService.getAMission(this.props.missionId, token);
    return <DoWorkPage instanceDetail={instanceDetail} missionDetail={missionDetail} token={token} readonly={false}/>
  };

  render() {
    return <AsyncComponent render={this.renderContent} componentWhenLoading={<Loading/>}/>;
  }
}
