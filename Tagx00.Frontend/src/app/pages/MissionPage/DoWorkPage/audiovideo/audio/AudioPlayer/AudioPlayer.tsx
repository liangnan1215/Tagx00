import React from 'react';

interface Props {
  url: string;
  onPlay?(): void;
  onPause?(): void;
  onTimeChanged?(offset: number): void;
  setRef?: (ref: HTMLAudioElement) => void;
}

interface State {

}

export class AudioPlayer extends React.Component<Props, State> {

  audioRef: HTMLAudioElement;

  onTimeUpdate = () => {
    this.props.onTimeChanged && this.props.onTimeChanged(this.audioRef.currentTime);
  };

  onPlay = () => {
    this.props.onPlay && this.props.onPlay();
  };

  onPause = () => {
    this.props.onPause && this.props.onPause();
  };

  ref = (ref) => {
    this.audioRef = ref;
    this.props.setRef && this.props.setRef(ref);
  };

  render() {
    return <audio
      ref={this.ref}
      controls
      onTimeUpdate={this.onTimeUpdate}
      onPlay={this.onPlay}
      onPause={this.onPause}
      src={this.props.url}
    />;
  }
}
