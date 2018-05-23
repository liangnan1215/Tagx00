import React, { ReactNode } from "react";
import { ImageMissionDetail, ImageMissionType } from "../../../../models/mission/image/ImageMission";
import { ImageInstanceDetail } from "../../../../models/instance/image/ImageInstanceDetail";
import { ImageNotation, ImageWorkStore } from "./ImageWorkStore";
import { observer } from "mobx-react";
import { ImagePartWorkPage } from "./ImagePartWorkPage";
import { ImageDistrictWorkPage } from "./ImageDistrictWorkPage";
import { ImageWholeWorkPage } from "./ImageWholeWorkPage";
import { message, Progress } from 'antd';
import { CompleteModal } from "../../../../components/ImageWork/CompleteModal";
import { ImageJob } from "../../../../models/instance/image/job/ImageJob";
import { action, observable, runInAction } from "mobx";
import { WorkerService } from "../../../../api/WorkerService";
import { Inject, Module } from "react.di";
import { DistrictJob } from "../../../../models/instance/image/job/DistrictJob";
import { LocaleStore } from "../../../../stores/LocaleStore";

interface Props {
  instanceDetail: ImageInstanceDetail;
  missionDetail: ImageMissionDetail;
  jumpBack: () => void;
  readonlyMode: boolean;
}

const ID_PREFIX = "drawingPad.common.";


@Module({
  providers: [
    {provide: ImageWorkStore, useClass: ImageWorkStore, noSingleton: true}
  ]
})
@observer
export class ImageWorkPage extends React.Component<Props, {}> {

  @Inject store: ImageWorkStore;
  @Inject localeStore: LocaleStore;

  @observable finishModalShown = true;

  @action saveWork = async (notation: ImageNotation) => {
    this.store.saveWork(notation);
    await this.store.saveProgress();
    message.success(this.localeStore.get(ID_PREFIX + "finish.workSaved"));
  };

  goNext = (notation: ImageNotation) => {
    this.store.saveWork(notation);
    this.store.nextWork();
  };

  goPrevious = () => {
    this.store.previousWork();
  };

  componentDidMount() {
    const {instanceDetail, missionDetail} = this.props;
    this.store.initialize(missionDetail, instanceDetail);
    this.forceUpdate();
  }

  @action componentDidUpdate() {
    if (this.store.finished) {
      if (this.props.readonlyMode) {
        message.info(this.localeStore.get(ID_PREFIX + "finish.readonlyComplete"));
        this.store.workIndex--;
        return;
      } else {
        this.finishModalShown = true;
      }
    }
  }

  submit = async () => {

    const result = await this.store.submit();
    if (result) {
      console.log("success");
      this.props.jumpBack();
    } else {
      console.log("failure");
    }
  };

  @action cancelFinishModal = () => {
    this.store.workIndex--;
    this.finishModalShown = false;
  };

  saveProgress = async () => {
    await this.store.saveProgress();
    this.props.jumpBack();
  };


  chooseWorkPage() {
    const currentWork: ImageNotation = this.store.currentWork;

    if (this.store.finished) {
      if (this.props.readonlyMode) {
        return <div/>;
      } else {
        return <CompleteModal shown={this.finishModalShown}
                              submit={this.submit}
                              saveProgress={this.saveProgress}
                              goBack={this.cancelFinishModal}

        />;
      }

    }

    const params = {
      notation: currentWork as any,
      submit: this.saveWork,
      missionDetail: this.props.missionDetail,
      goNext: this.goNext,
      controllerProps: {
        goPrevious: this.goPrevious,
        previousAvailable: this.store.workIndex != 0,
        saving: this.store.saving
      },
      readonlyMode: this.props.readonlyMode
    };

    switch (currentWork.job.type) {
      case ImageMissionType.PART:
        return <ImagePartWorkPage {...params}/>;
      case ImageMissionType.DISTRICT:
        return <ImageDistrictWorkPage {...params}/>;
      case ImageMissionType.WHOLE:
        return <ImageWholeWorkPage  {...params}/>;
    }
  }

  render() {

    const {instanceDetail, missionDetail} = this.props;

    return <div style={{overflow: "hidden"}}>
      {this.store.initialized ? this.chooseWorkPage() : null}
      <div>
        <Progress percent={this.store.progress / this.store.totalCount * 100}
                  status="active"
                  format={percent => `${this.store.progress}/${this.store.totalCount}`}
        />
      </div>
    </div>;
  }
}
