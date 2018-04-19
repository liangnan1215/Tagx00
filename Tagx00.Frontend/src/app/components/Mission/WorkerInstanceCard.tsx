import React from "react";
import { Card, Popconfirm } from 'antd';
import { Instance } from "../../models/instance/Instance";
import { ImageMissionDetail } from "../../models/mission/image/ImageMission";
import { AsyncComponent } from "../../router/AsyncComponent";
import { MissionInstanceState } from "../../models/instance/MissionInstanceState";
import { RouterStore } from "../../stores/RouterStore";
import { UserStore } from "../../stores/UserStore";
import { Inject } from "react.di";
import { MissionService } from "../../api/MissionService";
import { CardAction, stubCard, truncateText } from "./util";
import { InstanceStateIndicator } from "./InstanceStateIndicator";
import { LocaleMessage } from "../../internationalization/components";
import { WorkerService } from "../../api/WorkerService";

const {Meta} = Card;


interface Props {
  instance: Instance;
  refresh(): void;
}

const ID_PREFIX = "missions.worker.myMissions.";

export class WorkerInstanceCard extends React.Component<Props, any> {

  @Inject userStore: UserStore;
  @Inject routerStore: RouterStore;

  @Inject missionService: MissionService;
  @Inject workerService: WorkerService;

  goToDoMission = () => {
    const missionId = this.props.instance.missionId;
    this.routerStore.jumpTo(`/mission/worker/${missionId}/doWork`);
  };

  abandonMission = async () => {
    const missionId = this.props.instance.missionId;
    await this.workerService.abandonMission(missionId, this.userStore.token);
    this.props.refresh();
  };

  goDetail = () => {
    const missionId = this.props.instance.missionId;
    this.routerStore.jumpTo(`/mission/worker/${missionId}`);
  };

  title(title: string) {
    return <div>
      <span style={{marginRight: "4px"}}>{title}</span> <InstanceStateIndicator instance={this.props.instance}/>
    </div>
  }


  getActions = (instance: Instance) => {
    const buttons = [];

    switch (instance.missionInstanceState) {
      case MissionInstanceState.IN_PROGRESS:
        buttons.push(
          <CardAction key={"continue"} iconType={"edit"} onClick={this.goToDoMission}
                      hoverTextId={ID_PREFIX + "cardActions.continue"}/>,
          <Popconfirm title={<LocaleMessage id={ID_PREFIX + "cardActions.abandonConfirm"}/>} onConfirm={this.abandonMission}>
            <CardAction key={"delete"} iconType={"delete"}
                        hoverTextId={ID_PREFIX + "cardActions.abandon"}/>
          </Popconfirm>
        );
        break;
      case MissionInstanceState.SUBMITTED:
        break;
      case MissionInstanceState.ABANDONED:
        buttons.push(<CardAction key={"continue"} iconType={"edit"} onClick={this.goToDoMission}
                                 hoverTextId={ID_PREFIX + "cardActions.continue"}/>);
    }

    buttons.push(<CardAction key={"search"} iconType={"search"} onClick={this.goDetail}
                             hoverTextId={ID_PREFIX + "cardActions.seeMore"}/>);

    return buttons;

  };

  renderCard = async () => {
    const {instance} = this.props;
    const mission: ImageMissionDetail = await this.missionService.getAMission(instance.missionId, this.userStore.token);
    const {publicItem} = mission;
    return <Card
      style={{width: 300}}
      cover={<img alt="example" src={publicItem.coverUrl}/>}
      actions={this.getActions(instance)}>
      <Meta
        title={this.title(publicItem.title)}
        description={truncateText(publicItem.description)}
      />
    </Card>
  };

  render() {
    return <AsyncComponent render={this.renderCard} componentWhenLoading={stubCard}/>;
  }
}
