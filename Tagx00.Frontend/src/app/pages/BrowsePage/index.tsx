import React from "react";
import { observer } from "mobx-react";
import { BrowserStore } from "../../stores/BrowserStore";
import { Module } from "react.di";
import { BrowseAnimation } from "./BrowerAnimation";


@Module({
  providers: [
    BrowserStore
  ]
})
@observer
export class BrowsePage extends React.Component<any, any> {
  render() {
    return (
      <BrowseAnimation/>
    )
  }
}
