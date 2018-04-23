import React from "react";
import { observer } from "mobx-react";
import { Layout } from "antd";
import TweenOne from 'rc-tween-one';
import OverPack from 'rc-scroll-anim/lib/ScrollOverPack';
import QueueAnim from 'rc-queue-anim';
import { Inject } from "react.di";
import { BrowserStore } from "../../stores/BrowserStore";
import { SvgImg } from "../../components/Common/SvgImg";
import { SearchBar } from "./SearchBar";
import { BrowserMissionList } from "./BrowserMissionList";
import { action } from "mobx";
import ReactDOM from "react-dom";

const {Content} = Layout;

const standardBarMoveTop: number = -21;
const standardLogoMoveTop: number = -100;

@observer
export class BrowserAnimatedContent extends React.Component<any, any> {

  @Inject browserStore: BrowserStore;
  searchBar: React.RefObject<any> = React.createRef();

  @action handleResize = () => {
    console.log(ReactDOM.findDOMNode(this.searchBar.current))
    this.browserStore.searchBarWidth = -parseInt(this.searchBar.current.width);
    this.browserStore.resizeMoveRate();
  };

  componentDidMount() {
    window.onresize = this.handleResize;
  }

  componentWillUnmount() {
    window.onresize = null;
  }

  render() {
    const barMoveTop = standardBarMoveTop / this.browserStore.moveHeightRate + "%";
    const logoMoveTop = standardLogoMoveTop / this.browserStore.moveHeightRate + "%";
    const logoMoveLeft = this.browserStore.searchBarWidth;
    const contentAnimation = [
      {scale: 1},
      {scale: 1},
      {marginTop: barMoveTop + '%'},
    ];
    const logoAnimation = [
      {scale: 0.6},
      {marginLeft: logoMoveLeft},
      {scale: 0.3, marginTop: logoMoveTop + '%'}
    ];
    return <div>{!this.browserStore.isStop ? (
      <QueueAnim className="queue-demo"
      >
        <TweenOne animation={logoAnimation}
                  paused={this.browserStore.paused}
                  reverse={this.browserStore.reverse}
                  className="code-box-shape">
          <div style={{textAlign: 'center', marginBottom: '-10%', marginTop: '5%'}}>
            <SvgImg filePath={"tag_x00_logo.svg"} height={200} width={200}/>
          </div>
        </TweenOne>,
        <TweenOne animation={contentAnimation}
                  paused={this.browserStore.paused}
                  reverse={this.browserStore.reverse}
                  className="code-box-shape">
          <SearchBar ref={this.searchBar}/>
          <Content>
            <BrowserMissionList/>
          </Content>
        </TweenOne>
      </QueueAnim>
    ) : (<div>
      <div style={{textAlign: 'center', marginLeft: logoMoveLeft}}>
        <SvgImg filePath={"tag_x00_logo.svg"} height={60} width={60}/>
      </div>
      <div style={{marginTop: barMoveTop}}>
        <SearchBar ref={this.searchBar}/>
        <Content>
          <BrowserMissionList/>
        </Content>
      </div>
    </div>)}</div>
  }
}
