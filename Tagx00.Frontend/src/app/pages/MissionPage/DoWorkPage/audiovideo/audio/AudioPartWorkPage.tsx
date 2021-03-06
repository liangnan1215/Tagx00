import React from 'react';
import { AudioNotation, AudioWorkPageProps, AudioWorkPageState } from "./shared";
import { AudioPartJob, AudioPartTuple } from "../../../../../models/instance/audio/job/AudioPartJob";
import { AudioWholeJob } from "../../../../../models/instance/audio/job/AudioWholeJob";
import { AudioMissionType } from "../../../../../models/mission/audio/AudioMission";
import { toJS } from "mobx";
import { TagDescriptionTuple, TagTuple } from "../../../../../models/instance/TagTuple";
import { PartJobTuple } from "../../../../../models/instance/image/job/PartJob";
import { WorkPageLayout } from "../../WorkPageLayout";
import { AudioPlayer } from "../../../../../components/Mission/AudioPlayer";
import { MediaTupleList } from "../MediaTupleList";
import immer from "immer";
import { AudioMissionTipCard } from "../../../../../components/Mission/MissionTipCard/AudioMissionTipCard";
import { ProgressController } from "../../../../../components/Mission/WorkPageSuite/ProgressController";
import { AudioWorkPageLayout } from "./AudioWorkPageLayout";
import { TagDescriptionTuplePanel } from "../../../../../components/Mission/WorkPageSuite/TagDescriptionPanel";

function initializeNotation(notation: AudioNotation<AudioPartJob>) {
  if (!(notation.job && notation.job.tupleList)) {
    notation.job = {
      type: AudioMissionType.PART,
      tupleList: []
    };
  }
  return notation;
}

interface Props extends AudioWorkPageProps<AudioPartJob> {

}

interface State extends AudioWorkPageState<AudioPartJob> {
  selected: AudioPartTuple;
}

export class AudioPartWorkPage extends React.Component<Props, State> {

  state = {
    notation: initializeNotation(this.props.notation),
    selectedIndex: -1,
    selected: null,
    addingMode: false,
    width: 1,
    height: 1,
  };

  audioRef: AudioPlayer;

  static getDerivedStateFromProps(nextProps, prevState) {
    if (nextProps.notation !== prevState.notation) {
      return {
        notation: initializeNotation(nextProps.notation),
        selected: null
      }
    } else {
      return null;
    }
  }

  saveProgress = () => {
    console.log(toJS(this.state.notation));
    this.props.submit(this.state.notation);
  };
  //
  // onAudioProgress = (offset: number) => {
  //   this.audioOffset = offset;
  // };

  onAddTuple = () => {
    this.state.notation.job.tupleList.push({
      startOffset: this.audioRef.currentTime,
      endOffset: this.audioRef.currentTime,
      tuple: {
        tagTuples: [],
        descriptions: []
      }
    });
    this.forceUpdate();
  };


  onRemoveTuple = (tuple: AudioPartTuple) => {
    this.state.notation.job.tupleList = this.state.notation.job.tupleList.filter(x => x !== tuple);
    this.forceUpdate();
  };

  onSelect = (tuple: AudioPartTuple) => {
    this.setState({selected: tuple});
  };

  onPlay = (tuple: AudioPartTuple) => {
    const audio = this.audioRef;

    audio.playRegion(tuple.startOffset, tuple.endOffset);
  };

  goNext = () => {
    this.props.goNext(this.state.notation);
  };

  onTupleChange = (tuple: TagDescriptionTuple) => {
    if (this.state.selected) {
      this.state.selected.tuple = tuple;
      this.forceUpdate();
    }
  };

  playOrPause = () => {
    this.audioRef.playOrPause();
  };


  setStartTime = (tuple: AudioPartTuple) => {
    tuple.startOffset = this.audioRef.currentTime;
    this.forceUpdate();
  };

  setEndTime = (tuple: AudioPartTuple) => {
    tuple.endOffset = this.audioRef.currentTime;
    this.forceUpdate();
  };

  render() {

    const {missionDetail} = this.props;
    const {notation} = this.state;
    const {job} = notation;
    return <AudioWorkPageLayout
      saveProgress={this.saveProgress}
      previous={this.props.controllerProps.goPrevious}
      next={this.goNext}
      playOrPause={this.playOrPause}
    >
      <>
        <AudioPlayer url={this.props.notation.audioUrl}
          // onTimeChanged={this.onAudioProgress}
                     setRef={ref => this.audioRef = ref}

        />
      </>
      <>
        <AudioMissionTipCard audioMissionType={job.type}
                             tags={missionDetail.publicItem.tags}
                             allowCustomTag={missionDetail.publicItem.allowCustomTag}
                             title={missionDetail.publicItem.title}
        />
        <MediaTupleList tuples={this.state.notation.job.tupleList}
                        selected={this.state.selected}
                        readonly={this.props.readonlyMode}
                        onAdd={this.onAddTuple}
                        onSelect={this.onSelect}
                        onRemove={this.onRemoveTuple}
                        onPlay={this.onPlay}
                        onSetStartTime={this.setStartTime}
                        onSetEndTime={this.setEndTime}
        />
        {this.state.selected &&
        <TagDescriptionTuplePanel tuple={this.state.selected.tuple}
                                  onChange={this.onTupleChange}
                                  readonlyMode={this.props.readonlyMode}
                                  allowCustomTag={missionDetail.publicItem.allowCustomTag}
                                  tags={missionDetail.publicItem.tags}
        />
        }
      </>
      <>
        <ProgressController {...this.props.controllerProps}
                            goNext={this.goNext}
                            readonlyMode={this.props.readonlyMode}
                            saveProgress={this.saveProgress}
        />
      </>
    </AudioWorkPageLayout>
  }
}
